"""Thin CPU qualification wrapper; no dataset or label access."""
from pathlib import Path
import json
import subprocess
import time


def main():
    root = Path(__file__).resolve().parents[2]
    experiment = Path(__file__).resolve().parent
    trajectory_path = experiment / "qualification/trajectory.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if trajectory["record_mode"] != "prospective" or trajectory["state_at_start"]["source_commit"] != commit:
        raise RuntimeError("Qualification requires the prospective frozen source commit")
    output = experiment / "qualification_initial_state.pt"
    report_path = experiment / "qualification_result.json"
    if output.exists() or report_path.exists():
        raise FileExistsError("Qualification outputs already exist; reconcile before retry")

    from molgap.k1_node_width256 import make_encoder as make_width256
    from molgap.qm9_neural_atom import make_encoder as make_reference
    from molgap.training_reproducibility import (
        atomic_json, atomic_torch_save, configure_fp32_determinism, sha256_file,
    )
    from molgap.v4_runtime import state_dict_sha256
    import torch

    wall_start, cpu_start = time.perf_counter(), time.process_time()
    runtime = configure_fp32_determinism(42)
    baseline = make_reference("neural_atom_k1")
    reference_hash = state_dict_sha256(baseline.state_dict())
    reference_count = sum(p.numel() for p in baseline.parameters())
    configure_fp32_determinism(42)
    explicit = make_reference("neural_atom_k1", hidden_channels=192)
    if state_dict_sha256(explicit.state_dict()) != reference_hash:
        raise RuntimeError("Reference default differs from explicit width192")
    del explicit, baseline
    configure_fp32_determinism(42)
    candidate = make_width256()
    state = {k: v.detach().cpu().clone() for k, v in candidate.state_dict().items()}
    count = sum(p.numel() for p in candidate.parameters())
    if reference_count != 3_658_817 or count != 6_035_201 or count > 2 * reference_count:
        raise RuntimeError("Unexpected parameter count or user capacity ceiling exceeded")
    if not all(torch.isfinite(v).all().item() for v in state.values()):
        raise RuntimeError("Initial tensor state is nonfinite")
    digest = state_dict_sha256(state)
    atomic_torch_save(output, state)
    restored = torch.load(output, map_location="cpu", weights_only=True)
    if state_dict_sha256(restored) != digest:
        raise RuntimeError("Saved initial state failed roundtrip identity")
    report = {
        "schema": "molgap-k1-width256-initialization-qualification-v1",
        "status": "CPU_INITIALIZATION_QUALIFIED", "source_commit": commit,
        "trajectory_id": trajectory["trajectory_id"], "action_id": "A001",
        "candidate_parameters": count, "reference_parameters": reference_count,
        "parameter_ratio": count / reference_count,
        "initial_state": output.relative_to(root).as_posix(),
        "initial_state_file_sha256": sha256_file(output),
        "initial_state_sha256": digest, "reference_default_state_sha256": reference_hash,
        "reference_default_matches_explicit192": True, "finite": True,
        "saved_state_roundtrip": True, "data_access": False, "label_access": False,
        "width_initialization_equivalence": "not_applicable_different_tensor_shapes",
        "runtime": runtime, "cpu_seconds": time.process_time() - cpu_start,
        "wall_seconds": time.perf_counter() - wall_start,
        "gpu_qualification": "pending", "training_submission": "not_performed",
    }
    atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
