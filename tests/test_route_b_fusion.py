import torch

from molgap.route_b_fusion import train_route_b_fusion


def _payload(path, dimension, offset=0.0):
    value = {}
    for role, rows in (("train", 16), ("validation", 8), ("test", 8)):
        source_idx = torch.arange(rows)
        target = torch.stack(
            (source_idx.float(), source_idx.float() + 1, source_idx.float() + 2),
            dim=1,
        ) / 10
        value[role] = {
            "source_idx": source_idx,
            "cid": [str(index) for index in range(rows)],
            "embeddings": torch.randn(rows, dimension) + offset,
            "targets": target,
        }
    torch.save(value, path)


def test_route_b_fusion_end_to_end_and_resume(tmp_path):
    paths = {}
    for name, dimension in (
        ("gps9", 7),
        ("schnet_primary", 5),
        ("schnet_augmented", 6),
    ):
        path = tmp_path / f"{name}.pt"
        _payload(path, dimension)
        paths[name] = path
    output = tmp_path / "output"
    first = train_route_b_fusion(
        paths, "minimal", output, epochs=1, batch_size=4, hidden=8
    )
    resumed = train_route_b_fusion(
        paths,
        "minimal",
        output,
        epochs=2,
        batch_size=4,
        hidden=8,
        resume=True,
    )
    assert first["candidate"] == "minimal"
    assert resumed["resume_supported"]
    assert (output / "best.pt").exists()
    assert (output / "last.pt").exists()
    assert len(torch.load(output / "last.pt", weights_only=False)["log"]) == 2


def test_route_b_fusion_modes(tmp_path):
    paths = {}
    for name, dimension in (
        ("gps9", 7),
        ("gps11_160", 4),
        ("schnet_primary", 5),
        ("schnet_augmented", 6),
    ):
        path = tmp_path / f"{name}.pt"
        _payload(path, dimension)
        paths[name] = path
    for mode in ("concat", "bounded_residual"):
        result = train_route_b_fusion(
            paths,
            "precision",
            tmp_path / mode,
            epochs=1,
            batch_size=4,
            hidden=8,
            fusion_mode=mode,
        )
        assert result["fusion_mode"] == mode
