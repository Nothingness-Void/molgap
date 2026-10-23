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
from test_family_training_session import SyntheticKernel, assert_equal, events, make_session


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


def test_k1_batch_renorm_family_kernel_resume_journey(tmp_path):
    from copy import deepcopy
    from pathlib import Path

    import numpy as np
    from torch import nn
    from torch_geometric.nn.norm import BatchNorm as PyGBatchNorm

    from molgap.experiment_execution import build_addon_declaration
    from molgap.family_training_data import DatasetSelection
    from molgap.family_training_kernel import FamilyKernel, load_family_model
    from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
    from molgap.k1_batch_renorm import BatchRenorm1d
    from molgap.qm9_neural_atom import make_encoder
    from molgap.training_reproducibility import (
        atomic_torch_save, capture_rng_state, restore_rng_state,
    )
    from molgap.v4_runtime import state_dict_sha256

    config = {"correct_shift": 1, "warmup_updates": 3905, "ramp_updates": 3905,
        "r_max": 3.0, "d_max": 5.0}
    scale_config = {**config, "correct_shift": 0}
    assert [key for key in config if config[key] != scale_config[key]] == ["correct_shift"]

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        base = make_encoder("neural_atom_k1")
    initial = {key: value.detach().cpu().clone() for key, value in base.state_dict().items()}
    keys = tuple(initial)
    parameter_count = sum(parameter.numel() for parameter in base.parameters())
    trainable_count = sum(parameter.numel() for parameter in base.parameters() if parameter.requires_grad)
    del base
    digest = state_dict_sha256(initial)
    initial_path = tmp_path / "k1-initial.pt"
    atomic_torch_save(initial_path, {"model_state": initial, "state_sha256": digest})
    addon = build_addon_declaration(("neural_atom_k1", "3"), "k1_batch_renorm",
        repo_root=Path(__file__).resolve().parents[1], config=config)
    recipe = FamilyRecipe("neural_atom_k1", "composed", 42, digest,
        DatasetSelection("synthetic.json", "2" * 64, False), 128, 0, 2,
        .0004, .00001, ScheduleSpec("cosine_step", 2), "final-ema",
        "python-epoch-shuffle", 2, target_mean_eV=0., target_std_eV=1.,
        ema_decay=.999, addons=(addon,),
        torch_version=str(torch.__version__).split("+", 1)[0])

    def restore_safe_rng(state):
        restored = deepcopy(state)
        name, values, position, has_gauss, cached_gaussian = restored["numpy"]
        restored["numpy"] = (name, np.asarray(values, dtype=np.uint32),
            position, has_gauss, cached_gaussian)
        restore_rng_state(restored)

    from molgap.experiment_family_workflow import tensor_safe_rng_state

    # Interior r/d values ensure a missing stop-gradient changes input backward.
    value = torch.tensor([[0., 1.], [1., 2.], [2., 3.]])
    # Update 3905 is still ordinary BN; update 7810 reaches both frozen caps.
    boundary = BatchRenorm1d(2, correct_shift=1)
    boundary.train()
    with torch.no_grad():
        boundary.num_batches_tracked.fill_(3904)
    batch_mean = value.mean(0)
    batch_var = value.var(0, unbiased=False)
    ordinary = (value - batch_mean) / torch.sqrt(batch_var + boundary.eps)
    torch.testing.assert_close(boundary(value), ordinary * boundary.weight + boundary.bias)
    assert boundary.num_batches_tracked.item() == 3905

    corrected = BatchRenorm1d(2, correct_shift=1)
    shiftless = BatchRenorm1d(2, correct_shift=0)
    shiftless.load_state_dict(corrected.state_dict(), strict=True)
    corrected.train()
    shiftless.train()
    with torch.no_grad():
        for module in (corrected, shiftless):
            module.running_mean.zero_()
            module.running_var.fill_(1.)
            module.weight.copy_(torch.tensor([1.25, -.75]))
            module.bias.copy_(torch.tensor([.1, -.2]))
            module.num_batches_tracked.fill_(7809)
    differentiable = value.clone().requires_grad_()
    old_mean, old_var = corrected.running_mean.clone(), corrected.running_var.clone()
    variance, mean = torch.var_mean(differentiable.detach(), dim=0, unbiased=False)
    inv_std = torch.rsqrt(old_var + corrected.eps)
    expected_r = (torch.sqrt(variance + corrected.eps) * inv_std).clamp(1 / 3, 3)
    expected_d = ((mean - old_mean) * inv_std).clamp(-5, 5)
    normalized = (differentiable - mean) * torch.rsqrt(variance + corrected.eps)
    expected_output = (normalized * expected_r + expected_d) * corrected.weight + corrected.bias
    probe = torch.tensor([[1., -.3], [.2, 1.1], [-.7, .4]])
    actual_output = corrected(differentiable)
    torch.testing.assert_close(actual_output, expected_output)
    scale_output = shiftless(value)
    torch.testing.assert_close(actual_output.detach() - scale_output,
        (expected_d * corrected.weight).expand_as(actual_output))
    assert corrected.num_batches_tracked.item() == 7810
    assert torch.all((expected_r > 1 / 3) & (expected_r < 3))
    assert torch.all(expected_d.abs() < 5)
    (actual_output * probe).sum().backward()
    weighted_grad = probe * corrected.weight.detach() * expected_r
    expected_input_grad = (weighted_grad - weighted_grad.mean(0)
        - normalized.detach() * (weighted_grad * normalized.detach()).mean(0)) / torch.sqrt(variance + corrected.eps)
    torch.testing.assert_close(differentiable.grad, expected_input_grad)
    torch.testing.assert_close(corrected.running_var,
        old_var * .9 + value.var(0, unbiased=True) * .1)
    capped = BatchRenorm1d(2)
    capped.num_batches_tracked.fill_(7809)
    large = value * 10
    large_norm = (large - large.mean(0)) / torch.sqrt(large.var(0, unbiased=False) + capped.eps)
    torch.testing.assert_close(capped(large), large_norm * 3 + 5)

    eval_bn = BatchRenorm1d(2)
    native_eval = nn.BatchNorm1d(2)
    native_eval.load_state_dict(eval_bn.state_dict(), strict=True)
    eval_bn.eval()
    native_eval.eval()
    counter = eval_bn.num_batches_tracked.clone()
    torch.testing.assert_close(eval_bn(value), native_eval(value))
    assert torch.equal(eval_bn.num_batches_tracked, counter)
    cumulative = BatchRenorm1d(2, momentum=None)
    native_cumulative = nn.BatchNorm1d(2, momentum=None)
    native_cumulative.load_state_dict(cumulative.state_dict(), strict=True)
    with torch.no_grad():
        cumulative.num_batches_tracked.fill_(3)
        native_cumulative.num_batches_tracked.fill_(3)
    cumulative.train()
    native_cumulative.train()
    torch.testing.assert_close(cumulative(value), native_cumulative(value))
    for name, tensor in cumulative.state_dict().items():
        torch.testing.assert_close(tensor, native_cumulative.state_dict()[name])

    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            kernel = FamilyKernel(recipe, initial_path, device="cpu")
            modules = [module for module in kernel.model.modules() if isinstance(module, BatchRenorm1d)]
            assert len(modules) == 18
            assert all(type(getattr(block, name)) is PyGBatchNorm
                for block in kernel.model.local_blocks for name in ("norm1", "norm3"))
            assert tuple(kernel.model.state_dict()) == keys
            assert sum(parameter.numel() for parameter in kernel.model.parameters()) == parameter_count
            assert sum(parameter.numel() for parameter in kernel.model.parameters()
                if parameter.requires_grad) == trainable_count
            for key, tensor in initial.items():
                assert torch.equal(kernel.model.state_dict()[key].cpu(), tensor)
            assert all(module.correct_shift == 1 for module in modules)

            # This is an enabled-correction synthetic boundary injection, not 3905 trained updates.
            for module in modules:
                module.num_batches_tracked.fill_(7809)
            from test_family_training_kernel import synthetic_batch
            batch = synthetic_batch()
            kernel.model.train()
            loss, absolute, rows = kernel.step(batch, check_finite=True)
            assert loss.ndim == 0 and torch.isfinite(loss) and torch.isfinite(absolute) and rows == 2
            assert all(module.num_batches_tracked.item() == 7810 for module in modules)
            assert kernel.optimizer.state and kernel.ema.update_calls == 1

            checkpoint = {"model": deepcopy(kernel.model.state_dict()),
                "optimizer": deepcopy(kernel.optimizer.state_dict()),
                "ema": deepcopy(kernel.ema.state_dict()),
                "rng_state": tensor_safe_rng_state(capture_rng_state())}
            checkpoint_path = tmp_path / "k1-resume.pt"
            atomic_torch_save(checkpoint_path, checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

            with torch.random.fork_rng(devices=[]):
                selected_model, _ = load_family_model(recipe, initial_path)
            selected_model.load_state_dict(checkpoint["model"], strict=True)
            assert_equal(selected_model.state_dict(), checkpoint["model"])
            restore_safe_rng(checkpoint["rng_state"])

            kernel.step(batch, check_finite=True)
            expected = {"model": deepcopy(kernel.model.state_dict()),
                "optimizer": deepcopy(kernel.optimizer.state_dict()),
                "ema": deepcopy(kernel.ema.state_dict()),
                "rng_state": tensor_safe_rng_state(capture_rng_state())}
            resumed = FamilyKernel(recipe, initial_path, device="cpu")
            resumed.model.load_state_dict(checkpoint["model"], strict=True)
            resumed.optimizer.load_state_dict(checkpoint["optimizer"])
            resumed.ema.load_state_dict(checkpoint["ema"])
            restore_safe_rng(checkpoint["rng_state"])
            resumed.step(batch, check_finite=True)
            assert_equal(resumed.model.state_dict(), expected["model"])
            assert_equal(resumed.optimizer.state_dict(), expected["optimizer"])
            assert_equal(resumed.ema.state_dict(), expected["ema"])
            assert_equal(tensor_safe_rng_state(capture_rng_state()), expected["rng_state"])
            kernel.close()
            resumed.close()
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
