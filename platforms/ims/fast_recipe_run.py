"""IMS driver for the named execution policy fast_amp_nondeterministic: the fast GPT-T engine on a frozen recipe's data and init.

Same item contract as family_recipe_run.py (flags, last_checkpoint.pt, pause_receipt.json + exit 2, completion_manifest.json,
run_provenance.json, selected_model.pt in the family key layout so platforms/local/pcqm_tail_holdout_eval.py scores it). The engine
is experiments/pcqm_fast_engine_3d_20261124/fast_gptrans.py: AMP FP16, non-deterministic kernels, fused AdamW, GPU-resident
batches, one pinned torch.compile mode when Triton exists. The recipe is the family recipe with its execution_policy block replaced
by the named policy plus an "engine" block; nothing else differs. The batch order is randperm(seed + pass), a function of the pass
index alone, so a resume needs no sampler state. Not bit-reproducible by design.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import signal
import sys
import time
from pathlib import Path

from family_recipe_run import PAUSED, START, TAIL_START, sha256

POLICY = "fast_amp_nondeterministic"
ENGINE = Path(__file__).resolve().parents[2] / "experiments/pcqm_fast_engine_3d_20261124"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--initial-state", type=Path, required=True)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--phase", choices=("train",), required=True)
    ap.add_argument("--no-preflight-receipt", action="store_true")
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--cpus", type=int, required=True)
    ap.add_argument("--accelerator")
    ap.add_argument("--execution-policy", required=True)
    ap.add_argument("--wall-s", type=float, default=0.0, help="pause after this many seconds from process start (0 = no limit)")
    ap.add_argument("--max-steps", type=int, default=0, help="pause at this absolute optimizer step (local dry runs)")
    ap.add_argument("--training-workers", type=int, default=0, help="unused: batches are built on the GPU")
    ap.add_argument("--ogb-zip", type=Path)
    args = ap.parse_args(argv)
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "completion_manifest.json").is_file():
        print("training already complete; nothing to do", flush=True)
        return 0
    if importlib.util.find_spec("ogb") is None and args.ogb_zip is not None:
        sys.path.append(str(args.ogb_zip))
    sys.path.insert(0, str(ENGINE))
    import torch
    import torch.nn.functional as F
    from fast_gptrans import FastBondLocalPairNormGPTrans, GPUResidentGraphStore
    from molgap.experiment_family_workflow import tensor_digest
    from molgap.family_training_recipe import ScheduleSpec
    from molgap.training_reproducibility import atomic_json, atomic_torch_save, validate_accelerator_request
    from molgap.v4_runtime import read_frozen_state_artifact

    record = json.loads(args.recipe.read_text(encoding="utf-8"))
    policy, eng = record["execution_policy"], record["engine"]
    if args.execution_policy != POLICY or policy["name"] != POLICY or record["selection"] != "final-ema" or record["batch_size"] != 128:
        raise ValueError("Recipe does not declare the fast_amp_nondeterministic policy with final-ema selection and batch 128")
    state, _ = read_frozen_state_artifact(args.initial_state, expected_state_sha256=record["initialization_sha256"])
    manifest_path = args.input_root / record["dataset"]["manifest"]
    if sha256(manifest_path) != record["dataset"]["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the recipe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = manifest["roles"]
    shards = {role: [args.input_root / a["file"] for a in manifest["assets"]["topology"] if a["role"] == role] for role in ("train", "development")}
    first, rows = roles["train"]["source_idx_start"], roles["train"]["rows"]
    if first + rows > TAIL_START or roles["development"]["source_idx_stop"] > TAIL_START or rows != record["train_rows"]:
        raise ValueError("Rows reach tail_holdout_v1 or differ from the recipe")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    gpu = torch.cuda.get_device_name(0)
    if args.accelerator:
        validate_accelerator_request(args.accelerator, gpu, hip_runtime=False)
    device = torch.device("cuda")
    train = GPUResidentGraphStore(shards["train"], device=device, expected_source_start=first, expected_source_stop=first + rows)
    dev = GPUResidentGraphStore(shards["development"], device=device, expected_source_stop=roles["development"]["source_idx_stop"])
    if tensor_digest(train.source_idx.cpu(), role="source_idx") != record["train_role_identity"] or len(train) != rows:
        raise ValueError("Decoded train rows differ from the recipe role identity")
    y = train.y.double()
    mean, std = float(y.mean()), float(y.std())
    if not all(math.isclose(o, e, rel_tol=0.0, abs_tol=1e-12) for o, e in ((mean, record["target_mean_eV"]), (std, record["target_std_eV"]))):
        raise ValueError(f"Decoded train target statistics differ from the recipe: {(mean, std)}")
    mean_t, std_t = torch.tensor(mean, dtype=torch.float32, device=device), torch.tensor(std, dtype=torch.float32, device=device)

    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    model = FastBondLocalPairNormGPTrans(use_3d=False, dropout=eng["dropout"], drop_path=eng["drop_path"]).to(device)
    model.load_family_state(state)
    torch.manual_seed(record["seed"])
    params = list(model.parameters())
    names = [n for n, _ in model.named_parameters()]
    if set(names) != set(model.state_dict()):
        raise ValueError("Engine model owns buffers; the EMA below covers parameters only")
    optimizer = torch.optim.AdamW(params, lr=record["learning_rate"], weight_decay=record["weight_decay"],
                                  betas=tuple(policy["adamw_betas"]), eps=policy["adamw_epsilon"], fused=True)
    scaler = torch.amp.GradScaler("cuda")
    decay = record["ema_decay"]
    shadow = [p.detach().clone() for p in params]
    forward, compiled = model, False
    if eng["compile_mode"] != "none" and importlib.util.find_spec("triton") is not None:  # lazy: the first steps compile; a failure falls back to eager
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 32  # one graph per (n_pad, edge bucket) shape
        forward, compiled = torch.compile(model, mode=eng["compile_mode"], dynamic=False), True
    schedule = ScheduleSpec(**record["schedule"])
    total, per_pass, lr_peak = record["optimizer_steps"], rows // record["batch_size"], record["learning_rate"]
    passes = -(-total // per_pass)
    checkpoint, log = out / "last_checkpoint.pt", out / "train_log.jsonl"
    if not (out / "run_provenance.json").is_file():
        atomic_json(out / "run_provenance.json", {
            "source_commit": args.source_commit, "recipe_sha256": sha256(args.recipe), "manifest_sha256": record["dataset"]["manifest_sha256"],
            "train_rows": rows, "train_row_range": [first, first + rows], "tail_holdout_v1_read": False, "official_valid_read": False,
            "test_read": False, "initial_state_sha256": sha256(args.initial_state), "gpu": gpu, "torch": torch.__version__,
            "batch_size": record["batch_size"], "precision": "amp-fp16", "tf32": False, "execution_policy": POLICY,
            "deterministic_algorithms": False, "optimizer": "fused-adamw", "compile_requested": eng["compile_mode"], "compile_used": compiled,
            "preflight_receipt": "waived" if args.no_preflight_receipt else "accepted"})

    st = dict(step=0, skips=0, nonfinite=0, carried_s=0.0, pass_s=0.0, loss_sum=0.0, loss_n=0)  # all that a resume restores besides tensors
    if checkpoint.is_file():
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        for target, value in zip(shadow, ck["ema"]):
            target.copy_(value)
        torch.set_rng_state(ck["rng_cpu"])
        torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        st.update(ck["state"])
        print("resumed at step", st["step"], flush=True)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))

    def elapsed() -> float:
        return st["carried_s"] + time.monotonic() - START

    def save() -> None:
        atomic_torch_save(checkpoint, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "ema": shadow,
            "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(), "state": {**st, "carried_s": elapsed()}})

    @torch.no_grad()
    def dev_mae(weights) -> float:
        backup = [p.detach().clone() for p in params]
        for p, w in zip(params, weights):
            p.copy_(w)
        model.eval()
        error = torch.zeros((), dtype=torch.float64, device=device)
        for start in range(0, len(dev), eng["eval_batch"]):
            batch = dev.get_batch(torch.arange(start, min(len(dev), start + eng["eval_batch"])), pad_to_n=eng["pad_to_n"], e_bucket=eng["edge_bucket"])
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(**{k: v for k, v in batch.items() if k != "y"}).float() * std_t + mean_t
            error += (pred - batch["y"]).abs().double().sum()
        for p, b in zip(params, backup):
            p.copy_(b)
        model.train()
        return float(error) / len(dev)

    def pause_now() -> bool:
        return stop[0] or bool(args.max_steps and st["step"] >= args.max_steps) or bool(args.wall_s and time.monotonic() - START >= args.wall_s)

    (out / "pause_receipt.json").unlink(missing_ok=True)
    model.train()
    last_dev = None
    while st["step"] < total:
        epoch = st["step"] // per_pass
        lr = schedule.pre_update_lr(epoch, lr_peak)
        for group in optimizer.param_groups:
            group["lr"] = lr
        order = torch.randperm(rows, generator=torch.Generator().manual_seed(record["seed"] + epoch))[: per_pass * record["batch_size"]].view(per_pass, -1)
        began, pass_s0 = time.monotonic(), st["pass_s"]
        loss_sum = torch.tensor(st["loss_sum"], dtype=torch.float32, device=device)
        while st["step"] < min(total, (epoch + 1) * per_pass):
            batch = train.get_batch(order[st["step"] - epoch * per_pass], pad_to_n=eng["pad_to_n"], e_bucket=eng["edge_bucket"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = forward(**{k: v for k, v in batch.items() if k != "y"})
            loss = F.l1_loss(pred.float(), (batch["y"] - mean_t) / std_t)
            scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, record["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            st["skips"] += scaler.get_scale() < scale
            with torch.no_grad():
                torch._foreach_mul_(shadow, decay)
                torch._foreach_add_(shadow, [p.detach() for p in params], alpha=1.0 - decay)
                finite = torch.isfinite(loss)
                loss_sum += torch.where(finite, loss.detach(), torch.zeros_like(loss))
                st["nonfinite"] += int(not finite)  # a skipped fp16 step may overflow the forward; the scaler already skipped it
            st["step"] += 1
            st["pass_s"] = pass_s0 + time.monotonic() - began
            pausing = pause_now() and st["step"] < min(total, (epoch + 1) * per_pass)  # a pass end first logs its row; the pause comes one step later
            if st["step"] % record["checkpoint_every"] == 0 or pausing:
                st["loss_sum"] = float(loss_sum)
                if not math.isfinite(st["loss_sum"]) or st["nonfinite"] > st["step"] // 10:
                    raise ValueError(f"Training diverged: loss sum {st['loss_sum']}, non-finite steps {st['nonfinite']}")
                save()
            if st["step"] % 500 == 0:
                print(f"[step] {st['step']}/{total} {st['pass_s']:.0f}s in pass {epoch + 1}", flush=True)
            if pausing:
                atomic_json(out / "pause_receipt.json", {"status": "PAUSED", "optimizer_steps": st["step"], "checkpoint_sha256": sha256(checkpoint)})
                print(f"PAUSED at step {st['step']}/{total}; run the same command to resume", flush=True)
                return PAUSED
        row = {"epoch": epoch + 1, "step": st["step"], "lr": lr, "train_mae_eV": float(loss_sum) / per_pass * std, "pass_s": round(st["pass_s"], 2),
               "scaler_skips": st["skips"], "nonfinite_steps": st["nonfinite"]}
        evaluated = time.monotonic()
        if (epoch + 1) % eng["eval_every"] == 0 or st["step"] >= total:
            last_dev = row["ema_dev_mae_eV"] = dev_mae(shadow)
        if (epoch + 1) % eng["live_dev_every"] == 0 or st["step"] >= total:
            row["dev_mae_eV"] = dev_mae([p.detach() for p in params])
        row["eval_s"] = round(time.monotonic() - evaluated, 2)
        row["elapsed_s"] = round(elapsed())
        st.update(pass_s=0.0, loss_sum=0.0)
        save()
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print("[pass]", json.dumps(row), flush=True)
    selected = {"model": FastBondLocalPairNormGPTrans.family_state_dict({n: s.detach().cpu() for n, s in zip(names, shadow)}),
                "weights": "ema", "optimizer_step": st["step"]}
    atomic_torch_save(out / "selected_model.pt", selected)
    atomic_json(out / "selected_eval.json", {"step": st["step"], "dev_mae_eV": last_dev})
    atomic_json(out / "completion_manifest.json", {
        "format": "molgap-ims-fast-recipe-run-v1", "status": "COMPLETE", "arm": record["mode"], "source_commit": args.source_commit,
        "recipe_sha256": sha256(args.recipe), "execution_policy": POLICY, "optimizer_steps": st["step"], "completed_passes": passes,
        "scaler_skips": st["skips"], "nonfinite_steps": st["nonfinite"], "selection": record["selection"],
        "selected_model_sha256": sha256(out / "selected_model.pt"), "last_checkpoint_sha256": sha256(checkpoint)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
