"""Synthetic checks that the 500K pair changes only the disagreement term."""
from copy import deepcopy
from types import SimpleNamespace
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from molgap import pcqm_composed_500k as composed
from molgap import pcqm_500k_v4_evidence as runner
from molgap import pcqm_gptrans_v4
from molgap.k1_pretrained_combo import objective
from molgap.screen_policy import canonical_fingerprint


class DropoutBatchNormRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(3)
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(3, 1)
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return self.head(self.dropout(self.bn(batch.x))).reshape(-1)


def test_pair_preserves_rng_bn_supervision_and_isolates_consistency_gradient(monkeypatch):
    torch.manual_seed(4201)
    initial = DropoutBatchNormRegressor().double().train()
    batch = SimpleNamespace(x=torch.randn(32, 3, dtype=torch.double),
                            y=torch.linspace(-2, 2, 32, dtype=torch.double))
    monkeypatch.setattr(pcqm_gptrans_v4, "_forward", lambda model, data: model(data))
    results = {}
    for arm in (composed.K1_MEAN2, composed.K1):
        model = deepcopy(initial)
        torch.manual_seed(4202)
        loss, terms = composed._objective(arm, model, batch, 0.25, 1.5)
        rng = torch.get_rng_state().clone()
        parameters = tuple(model.parameters())
        gradient = torch.autograd.grad(loss, parameters, retain_graph=True)
        disagreement_gradient = torch.autograd.grad(terms["disagreement"], parameters)
        assert model.calls == 2
        assert model.bn.num_batches_tracked.item() == 2
        assert terms["teacher_mse"] is None
        assert terms["disagreement"] > 0
        results[arm] = (model, loss.detach(), terms, rng, gradient, disagreement_gradient)

    control, candidate = results[composed.K1_MEAN2], results[composed.K1]
    assert torch.equal(control[3], candidate[3])
    assert torch.equal(control[0].bn.running_mean, candidate[0].bn.running_mean)
    assert torch.equal(control[0].bn.running_var, candidate[0].bn.running_var)
    assert torch.equal(control[2]["supervised_l1"], candidate[2]["supervised_l1"])
    assert torch.equal(control[2]["disagreement"], candidate[2]["disagreement"])
    torch.testing.assert_close(candidate[1] - control[1], 0.1 * candidate[2]["disagreement"])
    assert any(torch.count_nonzero(g) for g in candidate[5])
    for grad0, grad1, disagreement_grad in zip(control[4], candidate[4], candidate[5]):
        torch.testing.assert_close(grad1 - grad0, 0.1 * disagreement_grad,
                                   rtol=1e-10, atol=1e-12)


def test_arm_contracts_keep_shared_recipe_but_distinct_resume_identity():
    control = runner.scientific_contract(composed.K1_MEAN2)
    candidate = runner.scientific_contract(composed.K1)
    assert composed.K1_MEAN2 in runner.COMPOSED_ARMS
    assert {key for key in control if control[key] != candidate[key]} == {"arm", "loss_fingerprint"}
    assert canonical_fingerprint(control) != canonical_fingerprint(candidate)
    assert candidate["loss_fingerprint"] == "normalized-gap-two-pass-label-L1-plus0.1-disagreement-MSE"
    assert composed.INITIAL_FILES[composed.K1_MEAN2] == composed.INITIAL_FILES[composed.K1]
    assert composed.INITIAL_TENSORS[composed.K1_MEAN2] == composed.INITIAL_TENSORS[composed.K1]
    assert composed.make_ema(composed.K1_MEAN2, nn.Linear(1, 1)) is None
    assert [composed.schedule(composed.K1_MEAN2, epoch) for epoch in range(60)] == [
        composed.schedule(composed.K1, epoch) for epoch in range(60)]


def test_historical_objective_default_is_unchanged():
    first, second, target = torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([0.0])
    default, _ = objective(first, second, target, mode="pretrained_consistency")
    explicit, _ = objective(first, second, target, mode="pretrained_consistency", consistency_weight=0.1)
    assert torch.equal(default, explicit)
    assert default.item() == pytest.approx(1.6)


@pytest.mark.parametrize("weight", [True, -0.1, 0.2, float("nan"), float("inf")])
def test_objective_rejects_unfrozen_weights(weight):
    values = torch.ones(2)
    with pytest.raises(ValueError, match="consistency weight"):
        objective(values, values, values, mode="pretrained_consistency", consistency_weight=weight)


def test_resume_manifest_cannot_cross_ablation_arms(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "platforms/kaggle/run_legacy_500k_pair.py"
    spec = importlib.util.spec_from_file_location("legacy_ablation_resume", path)
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    monkeypatch.setattr(bootstrap, "digest", lambda file: hashlib.sha256(Path(file).read_bytes()).hexdigest())
    root = tmp_path / "checkpoints" / composed.K1_MEAN2
    root.mkdir(parents=True)
    manifest = root / "stage_manifest.json"
    manifest.write_text(json.dumps({"arm": composed.K1, "source_sha256": "a" * 64,
                                    "next_epoch": 3, "artifacts": {}}), encoding="utf-8")
    arm = {"arm_id": composed.K1_MEAN2, "resume": {"mount": "checkpoints",
           "manifest_sha256": bootstrap.digest(manifest), "source_sha256": "a" * 64, "next_epoch": 3}}
    with pytest.raises(RuntimeError, match="Resume cursor/source changed"):
        bootstrap.resolve_resume(tmp_path, arm)
