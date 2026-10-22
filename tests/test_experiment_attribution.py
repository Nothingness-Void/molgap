from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from molgap.experiment_attribution import (
    ATTRIBUTION_FORMAT, _compare_paired_errors, _optimal_constant_shift, analyze_family_pair,
)
from molgap.experiment_cli import main as cli_main
from molgap.fast_sandbox import run_sandbox
from molgap.models.substrate import ModelSubstrate
from molgap.training_reproducibility import sha256_file
from test_family_training_runner import contract_file, context
from test_family_training_session import SyntheticKernel, events, make_session


class _ToyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(32, 8)
        self.local_blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
        self.head = torch.nn.Linear(8, 1)

    def forward(self, batch):
        h = self.embed(batch.x[:, 0])
        h = self.local_blocks[0](h)
        # Pool by source_idx/batch
        out = torch.zeros(batch.num_graphs, h.shape[-1], dtype=h.dtype, device=h.device)
        out.index_add_(0, batch.batch, h)
        return self.head(out).squeeze(-1)


def test_substrate_replace_submodule_tracks_ownership_and_sandbox_gradients():
    def factory():
        model = _ToyBackbone()
        substrate = ModelSubstrate(model, "neural_atom_k1")
        replacement = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.SiLU(), torch.nn.Linear(8, 8))
        substrate.replace_submodule("custom_block0", "local_blocks.0", replacement)
        substrate.seal()
        return model

    report = run_sandbox(factory)
    assert report["status"] == "PASSED"
    assert "custom_block0" in report["addons"]
    assert report["addons"]["custom_block0"]["status"] == "PASSED"
    assert report["addons"]["custom_block0"]["trainable_parameters"] == 4


def test_verify_loaded_family_model_catches_disconnected_addon():
    class _DisconnectedReplacement(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dead = torch.nn.Parameter(torch.ones(4))

        def forward(self, x):
            return x

    model = _ToyBackbone()
    substrate = ModelSubstrate(model, "neural_atom_k1")
    substrate.replace_submodule("dead_addon", "local_blocks.0", _DisconnectedReplacement())
    substrate.seal()

    report = run_sandbox(lambda: model)
    assert report["status"] == "FAILED"
    assert "disconnected" in report["error"]["message"].lower() or "independent" in report["error"]["message"].lower()


def _write_native_arm(arm_dir, *, arm_id, family="neural_atom_k1", weights="live"):
    """Use the actual shared output writer/inspector, with tiny CPU-only updates."""
    from molgap.family_training_loaders import FamilyLoaders
    from molgap.family_training_session import FamilyTrainingSession
    from molgap.family_training_outputs import SessionOutputs
    from molgap.family_training_runner import PROFILES
    from molgap.gptrans_execution_tuning import ExecutionTuning

    session = make_session(family, development=True, schedule_kind="cosine_step", events=events())
    if weights == "live":
        recipe = replace(session.recipe, selection="best-development-live", ema_decay=None)
        data = session.loaders.data
        session.close()
        session = FamilyTrainingSession(SyntheticKernel(recipe),
            FamilyLoaders(recipe, data, ExecutionTuning(training_workers=0)),
            checkpoint=lambda *args: None, selected=lambda *args: None, pass_finished=lambda *args: None)
    arm_dir.parent.mkdir(parents=True, exist_ok=True)
    source = arm_dir.parent / (arm_id + "-contract")
    source.mkdir()
    try:
        path, record = contract_file(source, session)
        ctx = replace(context(family, path), arm_id=arm_id)
        outputs = SessionOutputs(arm_dir, ctx, contract=path, adapter=PROFILES[family],
            trajectory_id="synthetic-" + arm_id, train_role_identity=record["train_role_identity"])
        session.checkpoint, session.selected, session.pass_finished = outputs.checkpoint, outputs.selected, outputs.pass_finished
        session.batch_finished = outputs.batch_finished
        progress = session.run()
        report = outputs.complete(progress, runtime={"platform": ctx.platform, "account": ctx.account,
            "precision": "fp32", "source_commit": ctx.source_commit,
            "source_archive_sha256": ctx.source_archive_sha256},
            costs=[{"metric": "wall_seconds", "unit": "seconds", "value": 100.,
                "status": "measured", "semantics": "process_wall", "hardware": "synthetic-cpu"}])
        assert report["status"] == "MECHANICALLY_VERIFIED", report
        return arm_dir
    finally:
        session.close()


@pytest.fixture
def native_pair(tmp_path):
    return (_write_native_arm(tmp_path / "control", arm_id="reference"),
            _write_native_arm(tmp_path / "candidate", arm_id="candidate"))


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
@pytest.mark.parametrize("weights", ["live", "ema"])
def test_analyze_actual_family_output_and_cli_diagnose(tmp_path, family, weights):
    ctrl_dir = _write_native_arm(tmp_path / "control", arm_id="reference", family=family, weights=weights)
    cand_dir = _write_native_arm(tmp_path / "candidate", arm_id="candidate", family=family, weights=weights)
    out_dir = tmp_path / "diagnosis"
    report = analyze_family_pair(
        ctrl_dir,
        cand_dir,
        output_dir=out_dir,
        n_bootstrap=500,
        seed=42,
    )
    assert report["format"] == ATTRIBUTION_FORMAT
    assert report["status"] == "DIAGNOSED"
    assert report["alignment"]["rows"] == 3
    assert report["alignment"]["selection_weights"] == weights
    assert report["primary_metric"] == ("live_dev_metric" if weights == "live" else "ema_dev_metric")
    assert len(report["trace_dynamics"]["paired_curve"]) == 2
    assert report["cost"]["candidate_to_control_wall_ratio"] == 1.
    assert report["scientific_acceptance"] == report["replay_readiness"] == "NOT_EVALUATED"
    assert (out_dir / "analysis_summary.json").is_file()
    assert (out_dir / "attribution_report.md").is_file()
    if (family, weights) == ("neural_atom_k1", "live"):
        cli_out = tmp_path / "cli_diagnosis"
        exit_code = cli_main([
            "diagnose",
            "--control", str(ctrl_dir),
            "--candidate", str(cand_dir),
            "--output", str(cli_out),
            "--bootstrap-samples", "200",
        ])
        assert exit_code == 0
        assert (cli_out / "analysis_summary.json").is_file()
        assert (cli_out / "attribution_report.md").is_file()


def test_posthoc_shift_remains_descriptive_arithmetic():
    import numpy as np
    shift, mae = _optimal_constant_shift(np.array([.01, .02, .03]))
    assert shift == pytest.approx(-.02)
    assert mae == pytest.approx(.02 / 3)


@pytest.mark.parametrize("artifact", ["predictions", "contract", "trace", "selected_model"])
def test_diagnose_rejects_declared_hash_mismatch(native_pair, artifact):
    ctrl, cand = native_pair
    path = cand / "output_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"][artifact]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch|not pinned"):
        analyze_family_pair(ctrl, cand, n_bootstrap=10)


def test_diagnose_rejects_wrong_context_even_with_updated_file_hash(native_pair):
    ctrl, cand = native_pair
    pred_path = cand / "development_predictions.pt"
    payload = torch.load(pred_path, map_location="cpu", weights_only=True)
    payload["context"]["logical_run_id"] = "another-job"
    torch.save(payload, pred_path)
    path = cand / "output_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"]["predictions"]["sha256"] = sha256_file(pred_path)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="context/source/arm mismatch"):
        analyze_family_pair(ctrl, cand, n_bootstrap=10)


@pytest.mark.parametrize("status", ["missing", "estimated"])
def test_diagnose_preserves_unknown_or_estimated_cost_without_measured_ratio(native_pair, status):
    ctrl, cand = native_pair
    path = cand / "output_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["costs"][0].update(status=status, value=None if status == "missing" else 120.)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = analyze_family_pair(ctrl, cand, n_bootstrap=10)
    assert report["cost"]["candidate"]["process_wall_seconds"] == (None if status == "missing" else 120.)
    assert report["cost"]["candidate_to_control_wall_ratio"] is None
    assert report["cost"]["candidate_minus_control_wall_seconds"] is None


def test_zero_bootstrap_leaves_ci_unknown_and_never_passes_gates(native_pair, tmp_path):
    paired = _compare_paired_errors([.1, .2], [.01, .02], n_bootstrap=0)
    assert paired["lower_95_eV"] is paired["upper_95_eV"] is None
    assert not paired["passes_material_gate"] and not paired["passes_noninferiority_gate"]
    report = analyze_family_pair(*native_pair, n_bootstrap=0, output_dir=tmp_path / "no-bootstrap")
    assert report["primary_comparison"]["upper_95_eV"] is None
    assert "bootstrap disabled" in (tmp_path / "no-bootstrap/attribution_report.md").read_text()


def test_diagnose_rejects_selection_mismatch_and_wrong_curve_metric(native_pair, tmp_path):
    ema = _write_native_arm(tmp_path / "ema", arm_id="ema", weights="ema")
    with pytest.raises(ValueError, match="selection weights differ"):
        analyze_family_pair(native_pair[0], ema, n_bootstrap=10)
    with pytest.raises(ValueError, match="metric must match"):
        analyze_family_pair(*native_pair, primary_metric="ema_dev_metric", n_bootstrap=10)
