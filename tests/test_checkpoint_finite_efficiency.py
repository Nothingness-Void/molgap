"""Execution-only finite checks keep checkpoint/resume validation fail-closed."""
from copy import deepcopy

import pytest
import torch

from molgap.experiment_family_workflow import _finite_tree
from molgap.training_reproducibility import nonfinite_tensor_keys


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["model", "optimizer", "ema", "cursor"])
def test_all_checkpoint_tensor_branches_still_reject_nonfinite(field, bad):
    state = {"model": {"weight": torch.ones(3)}, "optimizer": {"state": {
        0: {"step": torch.tensor(1.), "exp_avg": torch.ones(3)}}},
        "ema": {"weight": torch.ones(3)}, "cursor": {"pending_absolute": [torch.tensor(1.)]}}
    state[field] = {"bad": [torch.tensor(bad)]}
    with pytest.raises(ValueError, match="Nonfinite checkpoint state: checkpoint." + field + ".bad"):
        _finite_tree(state, "checkpoint")


@pytest.mark.parametrize("host_first", [False, True])
def test_first_failure_keeps_tree_order_across_host_and_tensor(host_first):
    items = [("tensor", torch.tensor(float("inf"))), ("host", float("nan"))]
    if host_first:
        items.reverse()
    with pytest.raises(ValueError, match="checkpoint." + items[0][0] + "$"):
        _finite_tree(dict(items), "checkpoint")


def test_colliding_display_paths_and_list_labels_do_not_hide_failed_tensors():
    with pytest.raises(ValueError, match=r"checkpoint.a.b$"):
        _finite_tree({"a.b": torch.ones(1), "a": {"b": torch.tensor(float("nan"))}}, "checkpoint")
    with pytest.raises(ValueError, match=r"checkpoint.list$"):
        _finite_tree({"list": (torch.ones(1), [torch.tensor(float("inf"))])}, "checkpoint")


def test_no_success_cache_can_hide_mutated_optimizer_or_metrics():
    state = {"optimizer": {"exp_avg": torch.ones(3)}, "cursor": [torch.tensor(1.)]}
    _finite_tree(state, "checkpoint")
    state["optimizer"]["exp_avg"][1] = float("nan")
    with pytest.raises(ValueError, match=r"checkpoint.optimizer.exp_avg$"):
        _finite_tree(state, "checkpoint")
    state["optimizer"]["exp_avg"].fill_(0)
    state["cursor"][0].fill_(float("inf"))
    with pytest.raises(ValueError, match=r"checkpoint.cursor$"):
        _finite_tree(state, "checkpoint")


def test_cpu_scalar_checks_are_batched_by_dtype_without_precision_conversion(monkeypatch):
    state = {"large64": torch.tensor(1e100, dtype=torch.float64),
        "bad32": torch.tensor(float("nan")), "good32": torch.tensor(1., requires_grad=True),
        "integer": torch.tensor(3), "boolean": torch.tensor(True),
        "complex": torch.tensor(complex(1, float("inf"))), "empty": torch.empty(0)}
    before, calls = deepcopy(state), []
    original = torch.isfinite
    def check(value):
        calls.append((value.dtype, tuple(value.shape)))
        return original(value)
    monkeypatch.setattr(torch, "isfinite", check)
    assert nonfinite_tensor_keys(state) == ["bad32", "complex"]
    assert calls.count((torch.float32, (2,))) == 1
    assert (torch.float32, ()) not in calls
    assert torch.equal(state["large64"], before["large64"])
    assert state["good32"].grad is None


def test_alias_keys_and_metadata_keep_the_mapping_api_order():
    bad = torch.tensor(float("nan"))
    assert nonfinite_tensor_keys({"metadata": "fixture", "first": bad,
                                  "good": torch.tensor(1.), "alias": bad}) == ["first", "alias"]


def test_heterogeneous_finite_state_and_bytes_remain_unchanged(tmp_path):
    state = {"model": {"w": torch.randn(3, requires_grad=True)}, "optimizer": {"step": torch.tensor(4.)},
        "cursor": {"metrics": [torch.tensor(0., dtype=torch.float64), torch.tensor(2.)]},
        "rng": torch.tensor([1, 2], dtype=torch.uint8), "flags": torch.tensor([False, True]),
        "empty": torch.empty(0), "metadata": (None, 1, "finite", .25)}
    before = tmp_path / "before.pt"
    after = tmp_path / "after.pt"
    # A shared file name keeps Torch's archive prefix identical for byte comparison.
    target = tmp_path / "state.pt"
    torch.save(state, target)
    before.write_bytes(target.read_bytes())
    _finite_tree(state, "checkpoint")
    torch.save(state, target)
    after.write_bytes(target.read_bytes())
    assert before.read_bytes() == after.read_bytes()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native accelerator validation is separate")
@pytest.mark.parametrize("branch", ["model", "optimizer", "ema", "cursor"])
def test_native_gpu_nonfinite_still_reports_first_path(branch):
    state = {branch: {"good": torch.ones(3, device="cuda"),
        "bad": torch.tensor(float("nan"), device="cuda")}, "later_host": float("inf")}
    with pytest.raises(ValueError, match="checkpoint." + branch + ".bad$"):
        _finite_tree(state, "checkpoint")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native accelerator validation is separate")
def test_native_host_failure_before_gpu_failure_keeps_original_priority():
    with pytest.raises(ValueError, match=r"checkpoint.host$"):
        _finite_tree({"host": float("nan"), "gpu": torch.tensor(float("inf"), device="cuda")}, "checkpoint")
