"""Copied runtime/bundle cases only; legacy submission/audit APIs are out of scope."""
import subprocess
from pathlib import Path

import pytest
import torch

from molgap.v4_bundle import build_v4_source_bundle
from molgap.v4_runtime import (
    certify_numerical_repeatability,
    make_adamw_compat,
    normalized_source_sha256,
    sample_std_compat,
    torch_load_compat,
    validate_standard_source_bundle,
)


def test_legacy_torch_load_and_adamw_signatures(monkeypatch, tmp_path):
    calls = {}

    def old_load(path, map_location=None):
        calls["load"] = (path, map_location)
        return {"ok": True}

    monkeypatch.setattr(torch, "load", old_load)
    assert torch_load_compat(tmp_path / "artifact.pt", map_location="cpu") == {"ok": True}
    assert calls["load"][1] == "cpu"

    def old_adamw(params, lr=0.001, weight_decay=0.0, foreach=False):
        calls["adamw"] = {"lr": lr, "weight_decay": weight_decay, "foreach": foreach}
        return object()

    monkeypatch.setattr(torch.optim, "AdamW", old_adamw)
    parameter = torch.nn.Parameter(torch.ones(()))
    make_adamw_compat([parameter], lr=0.002, weight_decay=0.1, fused=False)
    assert calls["adamw"] == {"lr": 0.002, "weight_decay": 0.1, "foreach": False}
    with pytest.raises(RuntimeError, match="cannot satisfy a fused-AdamW"):
        make_adamw_compat([parameter], lr=0.002, fused=True)


def test_sample_std_matches_legacy_torch_api():
    values = torch.arange(101, dtype=torch.float64)
    assert torch.equal(sample_std_compat(values), values.std(unbiased=True))
    assert torch.equal(sample_std_compat(values, correction=0), values.std(unbiased=False))


def test_source_hash_normalizes_all_line_endings(tmp_path):
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"x = 1\ny = 2\n")
    crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert normalized_source_sha256(lf) == normalized_source_sha256(crlf)


def test_numerical_repeatability_accepts_only_bounded_delta():
    first = {"weight": torch.tensor([1.0, 2.0]), "count": torch.tensor([1])}
    second = {"weight": torch.tensor([1.0 + 1e-6, 2.0]), "count": torch.tensor([1])}
    evidence = certify_numerical_repeatability(
        losses=[0.5, 0.5 + 1e-8],
        states=[first, second],
        maximum_loss_delta=1e-7,
        maximum_parameter_delta=1e-5,
    )
    assert evidence["accepted"] is True
    assert evidence["state_sha256"][0] != evidence["state_sha256"][1]
    with pytest.raises(RuntimeError, match="exceeds numerical tolerance"):
        certify_numerical_repeatability(
            losses=[0.5, 0.5],
            states=[first, {"weight": torch.tensor([1.1, 2.0]), "count": torch.tensor([1])}],
            maximum_parameter_delta=1e-7,
        )


def test_source_bundle_is_reproducible_and_commit_bound(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    source = repo / "runner.py"
    source.write_bytes(b"print('v4')\r\n")
    subprocess.run(["git", "add", "runner.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()

    first = build_v4_source_bundle(
        repo_root=repo, relative_paths=["runner.py"],
        output_dir=tmp_path / "bundle-a", source_commit=commit,
    )
    second = build_v4_source_bundle(
        repo_root=repo, relative_paths=["runner.py"],
        output_dir=tmp_path / "bundle-b", source_commit=commit,
    )
    assert first["archive_sha256"] == second["archive_sha256"]
    validate_standard_source_bundle(
        Path(first["archive"]), first["archive_sha256"], commit
    )

    source.write_text("print('dirty')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be committed"):
        build_v4_source_bundle(
            repo_root=repo, relative_paths=["runner.py"],
            output_dir=tmp_path / "dirty", source_commit=commit,
        )
