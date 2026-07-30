import json

import torch

from molgap.pcqm_route_b_evaluation import verify_downloaded_fusion


def _write(path, value=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _artifact(path):
    from molgap.pcqm_route_b_training import sha256_file

    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def test_verify_downloaded_fusion(tmp_path):
    payload = tmp_path / "payload"
    checkpoints = payload / "checkpoints"
    payload_artifacts = {}
    for name in (
        "gps9",
        "gps11_160",
        "primary_schnet",
        "augmented_schnet",
    ):
        path = checkpoints / f"{name}_best.pt"
        _write(path)
        payload_artifacts[path.relative_to(payload).as_posix()] = _artifact(path)
    (payload / "manifest.json").write_text(
        json.dumps({"status": "complete", "artifacts": payload_artifacts})
    )

    fusion = tmp_path / "fusion"
    artifacts = {}
    for seed in (42, 43, 44):
        path = fusion / "augmented_schnet" / f"seed_{seed}" / "best.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"base_name": "augmented_schnet", "seed": seed}, path)
        artifacts[path.relative_to(fusion).as_posix()] = _artifact(path)
    selection = {
        "status": "complete",
        "selection_scope": "scaffold-development-only",
        "selected_base_identity": "augmented_schnet",
        "seeds": [42, 43, 44],
        "official_valid_metric_read": False,
    }
    selection_path = fusion / "development_selection.json"
    selection_path.write_text(json.dumps(selection))
    artifacts["development_selection.json"] = _artifact(selection_path)
    (fusion / "completion_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "selected_base_identity": "augmented_schnet",
                "artifacts": artifacts,
            }
        )
    )

    checkpoint_paths, fusion_paths, accepted = verify_downloaded_fusion(
        payload_dir=payload, fusion_dir=fusion
    )
    assert set(checkpoint_paths) == {
        "gps9",
        "gps11_160",
        "primary_schnet",
        "augmented_schnet",
    }
    assert len(fusion_paths) == 3
    assert accepted["selected_base_identity"] == "augmented_schnet"
