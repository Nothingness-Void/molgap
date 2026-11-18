"""IMS driver for a frozen family recipe: standalone models (CutlassGT, GRIT-SPSE) on a fixed dataset.

It reuses the family kernel, loaders and session. It adds only what a V4 stage needs: atomic
last_checkpoint.pt, a SIGTERM/walltime pause, and preflight/completion receipts. Exit 2 means
"paused, run the same command again to resume". It has no Spec/package layer because that route
compiles only for platform "kaggle" (experiment_input.py).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import signal
import sys
import time
from pathlib import Path

PAUSED = 2
TAIL_START = 3_250_000  # tail_holdout_v1 begins here; AGENTS.md reserves it
START = time.monotonic()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--initial-state", type=Path, help="frozen state file; standalone models need none, they build from the recipe seed")
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--phase", choices=("preflight", "train"), required=True)
    ap.add_argument("--preflight", type=Path, help="train: the accepted preflight.json")
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--cpus", type=int, required=True)
    ap.add_argument("--accelerator", help="expected device name, for example A100")
    ap.add_argument("--wall-s", type=float, default=0.0, help="pause after this many seconds from process start (0 = no limit)")
    ap.add_argument("--max-steps", type=int, default=0, help="pause at this absolute optimizer step (local dry runs)")
    ap.add_argument("--ogb-zip", type=Path, help="pure-python ogb archive, used only when ogb is not importable")
    args = ap.parse_args(argv)
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == "train" and (out / "completion_manifest.json").is_file():
        print("training already complete; nothing to do", flush=True)
        return 0
    if importlib.util.find_spec("ogb") is None and args.ogb_zip is not None:
        sys.path.append(str(args.ogb_zip))

    import torch
    from molgap.experiment_family_workflow import tensor_digest
    from molgap.family_training_data import load_training_data
    from molgap.family_training_kernel import FamilyKernel
    from molgap.family_training_loaders import FamilyLoaders
    from molgap.family_training_policy import mode_policy
    from molgap.family_training_runner import _assert_target_statistics, read_recipe
    from molgap.family_training_session import FamilyTrainingSession
    from molgap.gptrans_execution_tuning import ExecutionTuning
    from molgap.training_reproducibility import (
        atomic_json, atomic_torch_save, build_runtime_manifest, configure_fp32_determinism, validate_accelerator_request)

    recipe_sha = sha256(args.recipe)
    record, recipe = read_recipe(args.recipe)
    if args.phase == "train":
        receipt = json.loads(args.preflight.read_text(encoding="utf-8"))
        if receipt.get("status") != "accepted" or receipt.get("recipe_sha256") != recipe_sha:
            raise ValueError("Preflight receipt is missing, rejected or for another recipe")
    # Data acceptance is CPU work and comes before any GPU model, as in the governed worker.
    policy = mode_policy(recipe.family, recipe.mode)
    data = load_training_data(args.input_root, recipe.dataset)
    if recipe.input_sidecar is not None or policy.attach_inputs is not None:
        data = policy.attach_inputs(args.input_root, recipe.input_sidecar, data)
    first, rows = data.train_source_start, data.train_rows
    if first + rows > TAIL_START or rows != recipe.train_rows:
        raise ValueError("Training rows reach tail_holdout_v1 or differ from the recipe")
    if tensor_digest(torch.arange(first, first + rows), role="source_idx") != record["train_role_identity"]:
        raise ValueError("Decoded train rows differ from the recipe role identity")
    _assert_target_statistics(recipe, data)
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    gpu = torch.cuda.get_device_name(0)
    if args.accelerator:
        validate_accelerator_request(args.accelerator, gpu, hip_runtime=False)
    initial = args.initial_state
    if initial is None:
        if not policy.standalone_model:
            raise ValueError("Only a standalone model may omit --initial-state")
        # The kernel still verifies a frozen state hash; this tiny fixed state carries it and is kept for scoring.
        initial = out / "initial_state_placeholder.pt"
        atomic_torch_save(initial, {"unused_standalone_model_builds_from_seed": torch.zeros(1)})
    tuning = ExecutionTuning(independent_models=1, total_cpu_budget=args.cpus)
    torch.set_num_threads(tuning.cpu_threads)
    runtime = build_runtime_manifest(configure_fp32_determinism(recipe.seed))
    kernel = FamilyKernel(recipe, initial, device="cuda")
    total = recipe.optimizer_steps
    checkpoint = out / "last_checkpoint.pt"
    log = out / "train_log.jsonl"
    if args.phase == "train" and not (out / "run_provenance.json").is_file():
        atomic_json(out / "run_provenance.json", {
            "source_commit": args.source_commit, "recipe_sha256": recipe_sha, "recipe_identity": recipe.identity,
            "manifest_sha256": recipe.dataset.manifest_sha256, "train_rows": rows, "train_row_range": [first, first + rows],
            "tail_holdout_v1_read": False, "official_valid_read": False, "test_read": False,
            "initial_state_sha256": sha256(initial), "gpu": gpu, "torch": torch.__version__,
            "runtime_fingerprint": runtime["runtime_fingerprint"], "batch_size": recipe.batch_size, "precision": "fp32"})

    def on_selected(progress, weights, state, _evaluation):
        atomic_torch_save(out / "selected_model.pt", {"model": state, "weights": weights, "optimizer_step": progress.optimizer_step})

    def on_pass(progress, metrics):
        lr = (metrics.get("learning_rates_used") or [recipe.learning_rate])[0]
        row = {"epoch": progress.pass_index, "step": progress.optimizer_step, "lr": lr, "elapsed_s": round(time.monotonic() - START),
               "train_mae_eV": metrics["normalized_train_gap_mae"] * recipe.target_std_eV}
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print("[pass]", json.dumps(row), flush=True)

    last = [time.monotonic(), 0]

    def on_batch(progress, *, learning_rates):
        if progress.optimizer_step % 500 == 0:
            now = time.monotonic()
            print(f"[step] {progress.optimizer_step}/{total} {(progress.optimizer_step - last[1]) / (now - last[0]):.1f} step/s", flush=True)
            last[:] = [now, progress.optimizer_step]

    session = FamilyTrainingSession(kernel, FamilyLoaders(recipe, data, tuning), checkpoint=lambda _p, state: atomic_torch_save(checkpoint, state),
                                    selected=on_selected, pass_finished=on_pass, batch_finished=on_batch)
    if checkpoint.is_file():
        session.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        print("resumed at step", session.progress.optimizer_step, flush=True)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    limit = 2 if args.phase == "preflight" else args.max_steps
    seen = [session.progress.optimizer_step]

    def should_pause():  # called after every acknowledged update, before its checkpoint hook
        seen[0] += 1
        return stop[0] or bool(limit and seen[0] >= limit) or bool(args.wall_s and time.monotonic() - START >= args.wall_s)

    (out / "pause_receipt.json").unlink(missing_ok=True)
    progress = session.run(should_pause=should_pause)
    if args.phase == "preflight":
        atomic_json(out / "preflight.json", {"status": "accepted", "recipe_sha256": recipe_sha, "optimizer_steps": progress.optimizer_step,
                                             "manifest_sha256": recipe.dataset.manifest_sha256, "gpu": gpu, "torch": torch.__version__})
        return 0
    if progress.status != "complete":
        atomic_json(out / "pause_receipt.json", {"status": "PAUSED", "optimizer_steps": progress.optimizer_step,
                                                 "checkpoint_sha256": sha256(checkpoint)})
        print(f"PAUSED at step {progress.optimizer_step}/{total}; run the same command to resume", flush=True)
        return PAUSED
    atomic_json(out / "completion_manifest.json", {
        "format": "molgap-ims-family-recipe-run-v1", "status": "COMPLETE", "arm": recipe.mode, "source_commit": args.source_commit,
        "recipe_sha256": recipe_sha, "optimizer_steps": progress.optimizer_step, "sample_presentations": progress.sample_presentations,
        "completed_passes": progress.pass_index, "selection": recipe.selection,
        "selected_model_sha256": sha256(out / "selected_model.pt"), "last_checkpoint_sha256": sha256(checkpoint)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
