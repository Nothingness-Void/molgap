"""Synthetic checks only; no cache/checkpoint inference or local training."""
import json
from pathlib import Path

import pytest
import torch

from molgap.gptrans import OGBGPTransTiny
from molgap.gptrans_author_variants import apply_author_variant
from molgap.gptrans_capacity import construct as local_construct
from molgap.gptrans_local_control import (
    MODE, BASE_MODE, CAP, INSERTIONS, INITIAL_TENSOR_SHA256, PARAMETER_CAP,
    ControlledBondLocalBlock, bound_update, construct, freeze_initial,
    load_initial, architecture_identity,
)
from molgap.pcqm_gptrans_v4 import _state_sha256, _verify_model_identity, _ema_decay
from molgap.training_reproducibility import atomic_torch_save


def base_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return apply_author_variant(OGBGPTransTiny(), "degree_scale").state_dict()


def test_exact_frozen_tensors_parameter_count_depth_and_rng(tmp_path):
    base = base_state()
    before = torch.get_rng_state().clone()
    model, original = construct(MODE, base), local_construct(BASE_MODE, base)
    assert torch.equal(torch.get_rng_state(), before)
    assert sum(p.numel() for p in model.parameters()) == PARAMETER_CAP
    assert _state_sha256(model) == _state_sha256(original) == INITIAL_TENSOR_SHA256
    assert [i for i, b in enumerate(model.blocks, 1) if isinstance(b, ControlledBondLocalBlock)] == list(INSERTIONS)
    for a, b in zip(model.parameters(), original.parameters()):
        assert torch.equal(a, b)
    path, target = tmp_path / "base.pt", tmp_path / "control.pt"
    atomic_torch_save(path, {"model_state": base})
    freeze_initial(MODE, path, target)
    restored = load_initial(MODE, target)
    assert torch.equal(torch.get_rng_state(), before)
    assert _verify_model_identity(restored) == (PARAMETER_CAP, architecture_identity(MODE))
    assert _ema_decay(MODE) == .999
    payload = torch.load(target, weights_only=True)
    payload["variant"] = BASE_MODE
    atomic_torch_save(target, payload)
    with pytest.raises(ValueError, match="identity"):
        load_initial(MODE, target)


def test_bound_padding_virtual_permutation_and_batch_independence():
    node, update = torch.randn(2, 5, 8), torch.randn(2, 5, 8) * 10
    padding = torch.tensor([[[[False, False, False, True, True]]],
                            [[[False, False, False, False, False]]]])
    actual, (hn, un, scale) = bound_update(node, update, padding)
    assert torch.all(torch.linalg.vector_norm(actual, dim=(1, 2)) <= CAP * hn + 1e-6)
    assert torch.equal(actual[:, 0], torch.zeros_like(actual[:, 0]))
    assert torch.equal(actual[0, 3:], torch.zeros_like(actual[0, 3:]))
    assert torch.all((scale >= 0) & (scale <= 1))
    alone, _ = bound_update(node[:1, :3], update[:1, :3], padding[:1, ..., :3])
    assert torch.allclose(alone, actual[:1, :3], atol=1e-7, rtol=1e-6)
    changed = node.clone()
    changed[0, 0] = 1e5
    changed[0, 3:] = 1e5
    changed[1] *= 10
    other, _ = bound_update(changed, update, padding)
    assert torch.equal(other[0], actual[0])
    order = torch.tensor([0, 2, 1, 4, 3])
    permuted, _ = bound_update(node[:, order], update[:, order], padding[..., order])
    assert torch.allclose(permuted, actual[:, order], atol=1e-7, rtol=1e-6)
    assert not torch.equal(actual, update)


@pytest.mark.parametrize("node_zero,update_zero", [(False, True), (True, False), (True, True), (False, False)])
def test_zero_states_and_finite_gradients(node_zero, update_zero):
    node = (torch.zeros(2, 4, 8) if node_zero else torch.randn(2, 4, 8)).requires_grad_()
    update = (torch.zeros(2, 4, 8) if update_zero else torch.randn(2, 4, 8) * 10).requires_grad_()
    padding = torch.zeros(2, 1, 1, 4, dtype=torch.bool)
    bounded, _ = bound_update(node, update, padding)
    assert torch.isfinite(bounded).all()
    if node_zero or update_zero:
        assert torch.count_nonzero(bounded) == 0
    (node + bounded).square().sum().backward()
    assert torch.isfinite(node.grad).all() and torch.isfinite(update.grad).all()


def test_identity_forward_and_final_gap_gradient_connection():
    base = base_state()
    model, original = construct(MODE, base).eval(), local_construct(BASE_MODE, base).eval()
    x, attr = torch.zeros(6, 9, dtype=torch.long), torch.zeros(8, 3, dtype=torch.long)
    edge = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]])
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    args = (x, edge, attr, batch)
    assert torch.equal(model(*args), original(*args))
    with torch.no_grad():
        for block in model.blocks:
            block.output.weight.normal_(std=.03)
    for depth in INSERTIONS:
        model.blocks[depth-1]._capture_control_diagnostics = True
    loss = (model(*args).view(-1) - torch.tensor([.2, -.3])).abs().mean()
    loss.backward()
    for block in model.blocks:
        gradient = block.output.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0
    for depth in INSERTIONS:
        diag = model.blocks[depth-1]._control_diagnostics
        assert diag.shape == (3, 2) and torch.isfinite(diag).all()
        assert (diag[1] <= CAP + 1e-6).all()


def test_native_dispatch_and_closed_old_identity_unchanged(tmp_path):
    from molgap.experiment_spec import ADDONS
    from molgap.gptrans_author_screen import validate_arm_allocation
    from molgap.gptrans_capacity import architecture_identity as current_identity
    from molgap.v4_runtime import normalized_source_sha256
    from frozen_source_fixture import stage_frozen_sources, run_frozen_source
    assert ADDONS[(MODE, "1")].source_module == "molgap.gptrans_local_control"
    assert validate_arm_allocation({"arms": {MODE: {}}, "single_arm_reason": "one bounded hypothesis"}) == (MODE,)
    repo = Path(__file__).resolve().parents[1]
    bundle = json.loads((repo / "experiments/pcqm_gptrans_local_control_100k/reference/reference_bundle.json").read_bytes())
    acceptance = json.loads((repo / bundle["acceptance_ref"]).read_bytes())
    assert acceptance["source_archive_sha256"] == bundle["source_commit_or_archive"]
    # Closed identities belong to their accepted execution source. Legitimate
    # execution-only changes must not rewrite history or masquerade as old pins.
    snapshot = stage_frozen_sources(repo, acceptance["source_commit"], tmp_path / "frozen")
    result = run_frozen_source(snapshot, """
import json, sys
from molgap.gptrans_capacity import architecture_identity
print(json.dumps({"architecture_identity": architecture_identity(sys.argv[2])}))
""", BASE_MODE)
    assert result.returncode == 0, result.stderr
    historical = json.loads(result.stdout)["architecture_identity"]
    assert historical == bundle["architecture_config_identity"]
    changed = any(normalized_source_sha256(repo / "src/molgap" / name) !=
                  normalized_source_sha256(snapshot / "src/molgap" / name)
                  for name in ("gptrans_capacity.py", "gptrans.py"))
    if changed:
        assert current_identity(BASE_MODE) != historical
    else:
        assert current_identity(BASE_MODE) == historical
