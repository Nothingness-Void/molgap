import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from molgap.screen_policy import canonical_fingerprint
from molgap.v4_audit import audit_v4_paths
from molgap.v4_bundle import build_v4_source_bundle
from molgap.v4_runtime import (
    certify_numerical_repeatability,
    make_adamw_compat,
    normalized_source_sha256,
    sample_std_compat,
    torch_load_compat,
    validate_standard_source_bundle,
)
from molgap.v4_submission import (
    RUN_SPEC_FORMAT,
    bind_runtime_certificate,
    execute_run_spec,
    read_run_spec,
    runtime_calibration_fingerprint,
    validate_run_spec,
    write_run_spec,
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


def test_static_audit_detects_known_v4_failure_patterns(tmp_path):
    source = tmp_path / "bad_runner.py"
    source.write_text(
        "import torch\n"
        "x.std(correction=1)\n"
        "torch.optim.AdamW([], fused=True)\n"
        "torch.load('x', weights_only=False)\n"
        "torch.bincount(x)\n",
        encoding="utf-8",
    )
    findings = audit_v4_paths([source])["findings"]
    assert {item["rule"] for item in findings} == {
        "std-correction", "direct-adamw", "direct-torch-load", "accelerator-bincount"
    }

    aliased = tmp_path / "aliased_runner.py"
    aliased.write_text(
        "from torch import load as read_tensor\n"
        "from torch.optim import AdamW as Optimizer\n"
        "read_tensor('x')\n"
        "Optimizer([])\n",
        encoding="utf-8",
    )
    alias_rules = {item["rule"] for item in audit_v4_paths([aliased])["findings"]}
    assert alias_rules == {"direct-torch-load", "direct-adamw"}


def _valid_spec_and_certificate():
    certificate = {
        "format": "molgap-runtime-certificate-v1",
        "status": "accepted",
        "platform_id": "scnet-kunshan",
        "accelerator": "Hygon DCU",
        "precision": "fp32",
        "tf32_enabled": False,
        "deterministic_algorithms": True,
        "physical_batch_per_device": 128,
        "tail_batch_policy": "drop_last",
        "software_fingerprint": "a" * 64,
        "determinism_fingerprint": "b" * 64,
        "calibration_fixture_sha256": "c" * 64,
        "calibration_output_sha256": "d" * 64,
        "calibration_checks_passed": True,
        "runtime_fingerprint": "e" * 64,
    }
    contract = {
        "benchmark_id": "pcqm-gap-100k",
        "data_role_fingerprint": "train-dev-v1",
        "row_order_fingerprint": "row-order-v1",
        "feature_fingerprint": "ogb-categorical-v1",
        "target_fingerprint": "gap-ev-v1",
        "seed": 42,
        "precision": "fp32",
        "optimizer_fingerprint": "adamw-v1",
        "schedule_fingerprint": "cosine-v1",
        "loss_fingerprint": "l1-v1",
        "target_transform_fingerprint": "standardized-v1",
        "selection_fingerprint": "best-dev-v1",
        "role_access_fingerprint": "train-dev-only-v1",
        "sample_exposure": 1000,
        "tail_batch_policy": "drop_last",
    }
    spec = {
        "format": RUN_SPEC_FORMAT,
        "run_id": "gps-test-s42",
        "model_family": "gps",
        "model_id": "gps-variant-a",
        "architecture_sha256": "1" * 64,
        "runner_entrypoint": "molgap.test_adapter:run",
        "runner_parameters": {},
        "platform_id": certificate["platform_id"],
        "accelerator": certificate["accelerator"],
        "model_config": {"hidden_dim": 304, "layers": 9},
        "model_config_sha256": canonical_fingerprint({"hidden_dim": 304, "layers": 9}),
        "source": {"bundle_sha256": "f" * 64, "source_commit": "1234567"},
        "scientific_contract": contract,
        "physical_batch_per_device": 128,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
        "runtime_certificate_id": canonical_fingerprint(certificate),
    }
    certificate["runtime_calibration_fingerprint"] = runtime_calibration_fingerprint(spec)
    spec["runtime_certificate_id"] = canonical_fingerprint(certificate)
    spec["runtime_calibration_fingerprint"] = certificate[
        "runtime_calibration_fingerprint"
    ]
    return spec, certificate


def test_run_spec_separates_model_config_from_scientific_contract():
    spec, certificate = _valid_spec_and_certificate()
    result = validate_run_spec(spec, certificate)
    variant = {**spec, "model_config": {"hidden_dim": 320, "layers": 9}}
    variant["model_config_sha256"] = canonical_fingerprint(variant["model_config"])
    with pytest.raises(ValueError, match="calibration fingerprint"):
        validate_run_spec(variant, certificate)
    variant["runtime_certificate_id"] = None
    variant.pop("runtime_calibration_fingerprint")
    changed = validate_run_spec(variant, None)
    assert result["scientific_contract_sha256"] == changed["scientific_contract_sha256"]
    assert result["model_config_sha256"] != changed["model_config_sha256"]
    assert result["runtime_calibration_fingerprint"] != changed["runtime_calibration_fingerprint"]
    assert result["submission_payload_sha256"] != changed["submission_payload_sha256"]


def test_run_spec_rejects_batch_and_certificate_mismatch():
    spec, certificate = _valid_spec_and_certificate()
    bad_batch = {**spec, "physical_batch_per_device": 96}
    with pytest.raises(ValueError, match="physical batch exactly 128"):
        validate_run_spec(bad_batch, certificate)
    bad_cert = {**certificate, "tf32_enabled": True}
    with pytest.raises(ValueError, match="Runtime certificate mismatch"):
        validate_run_spec(spec, bad_cert)


def test_preflight_spec_can_be_prepared_before_platform_certificate(tmp_path):
    spec, _ = _valid_spec_and_certificate()
    spec.pop("runtime_certificate_id")
    prepared = write_run_spec(tmp_path / "run.json", spec)
    reloaded = read_run_spec(tmp_path / "run.json")
    assert prepared["submission_payload_sha256"] == reloaded["submission_payload_sha256"]
    assert reloaded["runtime_certificate_id"] is None

    template, certificate = _valid_spec_and_certificate()
    template.pop("runtime_certificate_id")
    template.pop("runtime_calibration_fingerprint")
    training = bind_runtime_certificate(
        template,
        certificate,
        runner_parameter_updates={"output": "/account/runs/train", "preflight_path": "/account/runs/preflight/preflight.json"},
    )
    assert training["runner_parameters"]["output"] == "/account/runs/train"
    assert training["runner_parameters"]["preflight_path"].endswith("preflight.json")
    assert training["runtime_certificate_id"] == canonical_fingerprint(certificate)


def test_shared_dispatch_passes_validated_config_to_family_adapter(monkeypatch):
    spec, _ = _valid_spec_and_certificate()
    spec.pop("runtime_certificate_id")
    spec.pop("runtime_calibration_fingerprint")
    module = types.ModuleType("v4_test_family_adapter")
    module.run = lambda **kwargs: {
        "mode": kwargs["mode"],
        "model_config": kwargs["run_spec"]["model_config"],
    }
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec["runner_entrypoint"] = "v4_test_family_adapter:run"
    result = execute_run_spec(spec, mode="preflight")
    assert result == {
        "mode": "preflight",
        "model_config": {"hidden_dim": 304, "layers": 9},
    }
    with pytest.raises(ValueError, match="requires an accepted platform"):
        execute_run_spec(spec, mode="train")
