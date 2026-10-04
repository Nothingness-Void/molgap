"""Extract a retained stage-only state; no model construction or execution."""
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from molgap.training_reproducibility import atomic_json, atomic_torch_save, sha256_file
from molgap.v4_runtime import state_dict_sha256

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
OUTPUT = ROOT / "platforms/_records/kaggle/training/k1_pretrained_combo_inputs"
PRETRAIN = Path("C:/Users/17449/.codex/worktrees/hierarchy-exposure/molgap/platforms/_records/kaggle/training/pcqm_pretraining_family_pair_release3/k1/pretraining")
INITIAL = Path("D:/w/k1-dropout-consistency/experiments/pcqm_k1_dropout_consistency/initial_state.pt")
TEACHER = Path("D:/w/k1-fusion-distillation/experiments/pcqm_k1_fusion_distillation/teacher_cache")


def main():
    completion = json.loads((PRETRAIN / "stage_complete.json").read_text())
    source = PRETRAIN / "last_checkpoint.pt"
    assert sha256_file(source) == completion["checkpoint_sha256"] == "aa41015dfd277c12618e17e3d438e3e5ecdb0fe05961e66372eba9f0046d7de0"
    # The accepted historical RNG contains only these known NumPy pickle globals.
    allowed = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype("uint32"))]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 9 and checkpoint["max_epochs"] == 10
    assert checkpoint["phase"] == "local_hierarchy" and checkpoint["seed"] == 42
    assert checkpoint["source_commit"] == completion["source_commit"]
    initial = torch.load(INITIAL, map_location="cpu", weights_only=True)
    assert state_dict_sha256(initial) == "8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd"
    state = checkpoint["model"]
    assert set(state) == set(initial)
    for key in state:
        assert state[key].shape == initial[key].shape and state[key].dtype == initial[key].dtype
    head = {key: value.detach().clone() for key, value in initial.items() if key.startswith("head.")}
    assert head
    prepared = {key: (head[key] if key in head else value.detach().clone()) for key, value in state.items()}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prepared_path = OUTPUT / "pretrained_initial_state.pt"
    atomic_torch_save(prepared_path, prepared)
    (EXP / "initialization_inputs").mkdir(exist_ok=True)
    shutil.copyfile(PRETRAIN / "stage_complete.json", EXP / "initialization_inputs/pretraining_stage_complete.json")
    teacher_destination = OUTPUT / "teacher_dataset"
    teacher_destination.mkdir(exist_ok=True)
    for name in ("manifest.json", "teacher_predictions.pt"):
        shutil.copyfile(TEACHER / name, teacher_destination / name)
    manifest_digest = sha256_file(teacher_destination / "manifest.json")
    teacher_manifest = json.loads((teacher_destination / "manifest.json").read_text())
    assert manifest_digest == "cd19a9154444f52721f901c7940caf7b1fb83c45a4946fcf8da02e218cce7504"
    assert sha256_file(teacher_destination / "teacher_predictions.pt") == teacher_manifest["sha256"]
    shutil.copyfile(TEACHER / "manifest.json", EXP / "initialization_inputs/teacher_manifest.json")
    receipt = {
        "prepared_state_path": str(prepared_path), "prepared_file_sha256": sha256_file(prepared_path),
        "initialization_sha256": state_dict_sha256(prepared),
        "pretrained_source_sha256": sha256_file(source), "head_reset_sha256": state_dict_sha256(head),
        "pretraining_source_commit": checkpoint["source_commit"], "pretraining_passes": 10,
        "pretraining_stage_complete_sha256": sha256_file(PRETRAIN / "stage_complete.json"),
        "pretraining_owner_commit": "8821b5ce893680121260627436dc11a8f7fd8403",
        "original_initial_state_file_sha256": sha256_file(INITIAL),
        "original_initial_state_tensor_sha256": state_dict_sha256(initial),
        "extraction": "checkpoint.model exact keys/shapes/dtypes; replace every head.* from original seed42 state; discard pretraining heads/optimizer/scheduler/RNG",
        "teacher_cache_manifest_sha256": manifest_digest, "teacher_identity": teacher_manifest["teacher_identity"],
        "teacher_dataset_directory": str(teacher_destination), "teacher_payload_sha256": teacher_manifest["sha256"],
        "historical_limits": ["Pretraining improvement and strict comparator qualification remain inconclusive.", "Historical pretraining device allocation is unknown; reused wall cost is not a new allocation.", "No teacher regeneration or scientific model execution occurred here."],
    }
    atomic_json(EXP / "initialization_provenance.json", receipt)
    atomic_json(teacher_destination / "dataset-metadata.json", {
        "id": "nothingnessvoid/molgap-k1-fusion-teacher-train100k-v1",
        "title": "MolGap K1 Fusion Teacher Train100K V1", "licenses": [{"name": "CC0-1.0"}],
    })
    print(json.dumps({"initialization_sha256": receipt["initialization_sha256"], "head_reset_sha256": receipt["head_reset_sha256"], "prepared_state_path": str(prepared_path)}))


if __name__ == "__main__":
    main()
