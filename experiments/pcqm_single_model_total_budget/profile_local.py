"""Short local execution diagnostic; no development data or model-quality claim."""
from __future__ import annotations

import gc
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace
import zipfile

from molgap.research_memory.schemas import validate_trajectory
from molgap.research_memory.trace import atomic_write, file_digest, json_bytes

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = Path("D:/molgap-data/pcqm_fixed_100k_v1")
OUT = Path("D:/molgap-artifacts/local_profiles/single_model_total_budget")
STEPS, WARMUP, BATCH, WALL_LIMIT = 20, 3, 128, 600
SHARD_SHA = "8cc4c6373c22c8f72ac302f514bb657c5b40a9a1c620c6228b760380ebec705e"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json_bytes(value))


def prepare(*, candidate_b=False):
    trajectory_name = "profile_b_trajectory.json" if candidate_b else "profile_trajectory.json"
    if (HERE / trajectory_name).exists():
        raise ValueError("Diagnostic already planned; review prior attempt before repeating")
    template = json.loads((ROOT / "experiments/pcqm_k1_hop_fusion_qualification/plan_input.json").read_text())
    prior = json.loads((ROOT / "experiments/pcqm_gptrans_bondlocal_pairnorm_lr_500k/runs/pcqm-gptrans-pairnorm-lr-500k-s42/prospective/pairnorm_lr1e3/rml_finalized/trajectory.json").read_text())
    trajectory = template["trajectory"]
    evidence = prior["result"]["evidence_ids"]
    rel = HERE.relative_to(ROOT).as_posix()
    source_files = [Path(__file__), ROOT / "src/molgap/gptrans_bond_local.py", ROOT / "src/molgap/gptrans_variants.py"]
    source = {p.relative_to(ROOT).as_posix(): file_digest(p) for p in source_files}
    trajectory.update(trajectory_id="TB-single-model-proportional-local-speed-20261028", family_id="gptrans-proportional-local-speed",
        question="What are the local speed and memory costs of 256/32/8 versus 384/48/12 with BondLocal, pair-update norm and EMA?")
    trajectory["hypothesis"].update(hypothesis_id="H-single-model-proportional-local-speed", observed_deficiency="Total-system savings cannot be inferred from parameter count.",
        supporting_evidence_ids=evidence, related_closed_family_ids=[prior["family_id"]],
        alternative_explanations=["The wider predictor is too slow or exceeds local memory", "Execution overhead dominates both dimensions"],
        changed_mechanism="Proportional node/pair/head dimensions only; no architecture-quality evaluation.",
        cheapest_falsifier="Two repeated real-training-input timing windows per shape, FP32 physical batch128, local RTX5060.",
        expected_native_cost_ref=f"{rel}/README.md", decision_changed_if_positive="Consider a separately authorized 500K proposal after measured resource checks.",
        decision_changed_if_negative="Do not run 500K merely because parameter count fits the ensemble budget.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    trajectory["state_at_start"].update(source_commit=commit, source_config_identity=file_digest(Path(__file__)),
        contract_refs=[f"{rel}/README.md"], reference_ids=evidence, prior_evidence_ids=evidence,
        parent_trajectory_ids=[prior["trajectory_id"]], role_snapshot_refs=[f"{rel}/README.md"], budget_snapshot_ref=f"{rel}/README.md")
    trajectory["actions"] = [{"action_id": "A001", "type": "local_execution_profile", "run_ids": ["local-proportional-profile"], "attempt_ids": ["attempt-001"], "source_commit": commit, "evidence_refs": [f"{rel}/README.md"], "cost_event_ids": []}]
    trajectory["result"] = {"evidence_ids": [], "evidence_refs": []}
    trajectory["decision"] = {"outcome": "ACTIVE", "decision_ref": f"{rel}/README.md", "next_allowed_actions": ["Local short timing only; total GPU wall limit600s"], "reopen_conditions": []}
    trajectory.update(comparison_class="CONTEXT_ONLY", comparison_readiness_ref=f"{rel}/README.md", comparison_blockers=["Execution diagnostic, not model-quality comparison"], reference_bundle_id=None)
    trajectory["profile_configuration"] = dict(batch=BATCH, warmup=WARMUP, measured=STEPS, order=[256,384,384,256], ema=.999, precision="fp32", maximum_gpu_wall_seconds=WALL_LIMIT, data_role="training only; prefix rows0:2944", source_files=source)
    if candidate_b:
        trajectory.update(trajectory_id="TB-single-model-depth16-local-speed-20261028",
                          question="Can the proposed16x256/32/8 arm B execute locally within the frozen resource envelope?")
        trajectory["hypothesis"]["hypothesis_id"] = "H-single-model-depth16-local-speed"
        trajectory["hypothesis"]["changed_mechanism"] = "Candidate B depth16; resource feasibility only, no accuracy or cross-run speed equivalence claim."
        trajectory["profile_configuration"].update(order=[256,256], layers=16)
    validate_trajectory(trajectory)
    save(HERE / trajectory_name, trajectory)
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_files:
            archive.write(path, path.relative_to(ROOT).as_posix())


def main(*, candidate_b=False):
    global OUT
    if candidate_b:
        OUT = OUT / "candidate_b"
    prepare(candidate_b=candidate_b)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Subset
    from molgap.gptrans_bond_local import make_bond_local_model
    from molgap.gptrans_execution_backend import bind_model_layout, prepare_layout, prepared_forward
    from molgap.gptrans_execution_hotpath import BoundForeachEMA
    from molgap.family_training_kernel import make_family_optimizer
    from molgap.family_training_core import optimizer_update
    from molgap.training_reproducibility import deterministic_memory_fill_context

    loader_path = ROOT / "experiments/pcqm_gptrans_pair_norm_500k_profile/run_profile.py"
    spec = importlib.util.spec_from_file_location("retained_gptrans_profile", loader_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    manifest = json.loads((DATA / "manifest.json").read_text())
    shard = DATA / "train/train_shard_0000.pt"
    entry = next(item for item in manifest["geometry_shards"] if item["file"] == "train/train_shard_0000.pt")
    if entry["role"] != "train" or entry["source_idx_min"] != 0 or file_digest(shard) != SHARD_SHA or entry["sha256"] != SHARD_SHA:
        raise ValueError("Training-input identity mismatch")
    dataset = helpers.load_shard(torch, shard)
    subset = Subset(dataset, range((WARMUP + STEPS) * BATCH))
    if not torch.cuda.is_available():
        raise ValueError("No local CUDA device; no CPU substitute")
    torch.set_num_threads(4)
    helpers.configure_torch(torch)
    started = time.perf_counter()
    report = dict(status="RUNNING", device=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        training_shard_sha256=SHARD_SHA, manifest_sha256=file_digest(DATA / "manifest.json"),
        quality_evaluation=False, development_labels_read=False, source_snapshot=str(OUT / "source.zip"), windows=[])
    save(OUT / "result.json", report)
    try:
        for width in ((256, 256) if candidate_b else (256, 384, 384, 256)):
            if time.perf_counter() - started > WALL_LIMIT:
                raise TimeoutError("Local timing budget exhausted")
            torch.cuda.empty_cache()
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            model = make_bond_local_model(node_channels=width, pair_channels=width//8,
                num_heads=width//32, num_layers=16 if candidate_b else 12, pair_update_norm=True)
            bind_model_layout(model)
            model = model.cuda().train()
            optimizer = make_family_optimizer(SimpleNamespace(family="gptrans_t", learning_rate=.001, weight_decay=.05), model.parameters())
            ema = BoundForeachEMA(model, .999)
            parameters = tuple(model.parameters())
            iterator = iter(helpers.make_loader(subset, {"workers": 0, "prefetch_factor": None}))
            samples = []
            for step in range(WARMUP + STEPS):
                if time.perf_counter() - started > WALL_LIMIT:
                    raise TimeoutError("Local timing budget exhausted")
                if step == WARMUP:
                    torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                began = time.perf_counter()
                batch = prepare_layout(next(iterator))
                idx = getattr(batch, "source_idx", getattr(batch, "row_index", None))
                if idx is None or not torch.equal(idx.view(-1).long(), torch.arange(step*BATCH, (step+1)*BATCH)):
                    raise ValueError("Training row order differs from authorized prefix")
                batch = batch.to("cuda", "x", "edge_index", "edge_attr", "batch", "y", "execution_layout", non_blocking=True)
                def loss_builder():
                    prediction = prepared_forward(model, batch)
                    target = (batch.y.view(-1) - 5.413434769952119) / 1.176728647311213
                    return F.l1_loss(prediction, target), None
                with deterministic_memory_fill_context(enabled=False):
                    loss, _ = optimizer_update(model, optimizer, loss_builder, clip_norm=1., check_finite=False,
                                               ema=ema, clip_parameters=parameters)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - began
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite timing loss")
                if step >= WARMUP:
                    samples.append(elapsed)
            item = dict(width=width, layers=16 if candidate_b else 12, pair_channels=width//8, heads=width//32,
                parameters=sum(p.numel() for p in parameters), weight_bytes=sum(p.numel()*p.element_size() for p in parameters),
                warmup=WARMUP, measured_steps=STEPS, batch=BATCH, median_step_seconds=statistics.median(samples),
                mean_step_seconds=statistics.mean(samples), molecules_per_second=BATCH/statistics.mean(samples),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                memory_scope="Complete model + AdamW + EMA + measured training activations; excludes other processes",
                timing_scope="Synchronized collation/layout/H2D/forward/backward/clip/optimizer/EMA; no checkpoint/evaluation")
            report["windows"].append(item)
            save(OUT / "result.json", report)
            print(json.dumps(item), flush=True)
            del loss, batch, model, optimizer, ema, parameters, iterator, loss_builder
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "COMPLETE"
    except (torch.OutOfMemoryError, TimeoutError) as exc:
        report.update(status="STOPPED", reason=type(exc).__name__ + ": " + str(exc))
    finally:
        report["gpu_allocation_wall_seconds"] = time.perf_counter() - started
        save(OUT / "result.json", report)
        save(HERE / ("profile_b_result.json" if candidate_b else "profile_result.json"), report)
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-b", action="store_true")
    raise SystemExit(main(candidate_b=parser.parse_args().candidate_b))
