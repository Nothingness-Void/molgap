import hashlib
import json

import torch

import molgap.pcqm_route_b_fusion as fusion


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(dimension):
    return {
        "model": {
            "head.0.weight": torch.eye(dimension),
            "head.0.bias": torch.zeros(dimension),
            "head.3.weight": torch.eye(dimension),
            "head.3.bias": torch.zeros(dimension),
            "head.5.weight": torch.zeros(3, dimension),
            "head.5.bias": torch.tensor([0.0, 0.0, 1.5]),
        },
        "target_mean_gap": 2.0,
        "target_std_gap": 0.5,
    }


def test_frozen_readout_and_bounded_correction(monkeypatch):
    for name in fusion.ENCODER_NAMES:
        monkeypatch.setitem(fusion.EXPECTED_DIMS, name, 4)
    checkpoint = _checkpoint(4)
    model = fusion.PCQMBoundedFusion(
        checkpoint,
        base_name="gps11_160",
        hidden=8,
        correction_scale_eV=0.10,
    )
    values = {name: torch.randn(7, 4) for name in fusion.ENCODER_NAMES}
    base = model.base_readout(values["gps11_160"])
    prediction = model(values)
    assert torch.allclose(base, torch.full((7,), 2.75))
    assert torch.all((prediction - base).abs() <= 0.100001)


def test_load_aligned_roles(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fusion,
        "EXPECTED_ROWS",
        {"train": 6, "dev": 4, "official": 2},
    )
    monkeypatch.setattr(
        fusion,
        "EXPECTED_PARTS",
        {"train": 2, "dev": 2, "official": 1},
    )
    for name in fusion.ENCODER_NAMES:
        monkeypatch.setitem(fusion.EXPECTED_DIMS, name, 4)
    encoder_dirs = {}
    role_rows = {"train": (3, 3), "dev": (2, 2)}
    for encoder_offset, name in enumerate(fusion.ENCODER_NAMES):
        root = tmp_path / name
        embedding_root = root / "embeddings"
        parts = []
        for role, rows_per_part in role_rows.items():
            (embedding_root / role).mkdir(parents=True, exist_ok=True)
            cursor = 0
            for part_index, rows in enumerate(rows_per_part):
                source_idx = torch.arange(cursor, cursor + rows)
                payload = {
                    "format": "molgap-pcqm-route-b-embedding-part-v1",
                    "name": name,
                    "role": role,
                    "source_shard": f"view/{role}_shard_{part_index:03d}.pt",
                    "source_idx": source_idx,
                    "embeddings": torch.full(
                        (rows, 4), float(encoder_offset), dtype=torch.float16
                    ),
                    "targets": source_idx.float() / 10,
                }
                path = embedding_root / role / f"{role}_{part_index:03d}.pt"
                torch.save(payload, path)
                parts.append(
                    {
                        "role": role,
                        "source_shard": payload["source_shard"],
                        "path": f"{role}/{path.name}",
                        "rows": rows,
                        "bytes": path.stat().st_size,
                        "sha256": _sha(path),
                    }
                )
                cursor += rows
        manifest = {
            "status": "complete",
            "name": name,
            "embedding_dim": 4,
            "parts": parts + [
                {
                    "role": "official",
                    "source_shard": "view/official_shard_200.pt",
                    "path": "official/unused.pt",
                    "rows": 2,
                    "bytes": 0,
                    "sha256": "unused",
                }
            ],
            "rows": fusion.EXPECTED_ROWS,
            "official_valid_metric_read": False,
        }
        (embedding_root / "manifest.json").write_text(json.dumps(manifest))
        torch.save(_checkpoint(4), root / "best.pt")
        encoder_dirs[name] = root

    payloads, report = fusion.load_aligned_roles(
        encoder_dirs, roles=("train", "dev")
    )
    assert report["roles"]["train"]["rows"] == 6
    assert payloads["gps9"]["train"].shape == (6, 4)
    assert torch.equal(
        payloads["gps9"]["dev_targets"], torch.tensor([0.0, 0.1, 0.2, 0.3])
    )

    consolidated_dir = tmp_path / "consolidated"
    manifest = fusion.export_consolidated_fusion_payload(
        encoder_dirs=encoder_dirs,
        output_dir=consolidated_dir,
    )
    assert manifest["status"] == "complete"
    loaded, loaded_report, checkpoints = (
        fusion.load_consolidated_fusion_payload(consolidated_dir)
    )
    assert loaded["augmented_schnet"]["train"].shape == (6, 4)
    assert loaded_report["roles"]["dev"]["rows"] == 4
    assert all(path.is_file() for path in checkpoints.values())
