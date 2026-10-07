import pytest
import torch

from molgap import k1_frozen_inference as inference


GOOD_SHA256 = "a" * 64
SOURCE_SHA256 = "b" * 64


def _checkpoint_state():
    return {
        "contract": {
            "arm": "k1_pretrained_consistency",
            "parameters": 3658817,
            "benchmark_id": "pcqm-composed500k-dev50k-60pass-v1",
            "precision": "fp32",
            "geometry_used": False,
            "teacher_used": False,
        },
        "epoch": 12,
        "source_sha256": SOURCE_SHA256,
        "mean": 0.25,
        "std": 1.5,
        "model": {"weight": torch.tensor([1.0])},
    }


def _patch_checkpoint_io(monkeypatch, state, *, digest=GOOD_SHA256, loader=None):
    from molgap import training_reproducibility, v4_runtime

    monkeypatch.setattr(training_reproducibility, "sha256_file", lambda path: digest)
    monkeypatch.setattr(
        v4_runtime,
        "torch_load_compat",
        loader or (lambda path, **kwargs: state),
    )


class _ParameterCount:
    def numel(self):
        return 3658817


class _FakeEncoder:
    def __init__(self):
        self.training = True
        self.loaded = None
        self.grad_enabled = True

    def load_state_dict(self, state, *, strict):
        assert strict is True
        self.loaded = state

    def parameters(self):
        return iter((_ParameterCount(),))

    def state_dict(self):
        return {"weight": torch.tensor([1.0])}

    def eval(self):
        self.training = False
        return self

    def requires_grad_(self, enabled):
        self.grad_enabled = enabled
        return self


def test_bad_checkpoint_sha_is_rejected_before_torch_load(monkeypatch):
    calls = []
    _patch_checkpoint_io(
        monkeypatch,
        None,
        digest="c" * 64,
        loader=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="bytes differ"):
        inference.load_native500k_k1(
            "synthetic.pt",
            expected_sha256=GOOD_SHA256,
            expected_source_sha256=SOURCE_SHA256,
            expected_epoch=12,
            checkpoint_kind="selected",
        )

    assert calls == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda state: state["contract"].update(precision="bf16"), "contract differs"),
        (lambda state: state.update(epoch=11), "epoch/source differs"),
        (lambda state: state.update(source_sha256="c" * 64), "epoch/source differs"),
    ],
    ids=("contract", "epoch", "source"),
)
def test_checkpoint_contract_epoch_and_source_bindings_are_enforced(
    monkeypatch, change, message
):
    state = _checkpoint_state()
    change(state)
    _patch_checkpoint_io(monkeypatch, state)
    monkeypatch.setattr(
        inference,
        "make_encoder",
        lambda name: pytest.fail("model construction must follow binding checks"),
    )

    with pytest.raises(ValueError, match=message):
        inference.load_native500k_k1(
            "synthetic.pt",
            expected_sha256=GOOD_SHA256,
            expected_source_sha256=SOURCE_SHA256,
            expected_epoch=12,
            checkpoint_kind="selected",
        )


def test_matching_synthetic_checkpoint_loads_frozen_eval_model(monkeypatch):
    state = _checkpoint_state()
    model = _FakeEncoder()
    _patch_checkpoint_io(monkeypatch, state)
    monkeypatch.setattr(inference, "make_encoder", lambda name: model)

    loaded, metadata = inference.load_native500k_k1(
        "synthetic.pt",
        expected_sha256=GOOD_SHA256,
        expected_source_sha256=SOURCE_SHA256,
        expected_epoch=12,
        checkpoint_kind="selected",
    )

    assert loaded is model
    assert model.loaded is state["model"]
    assert model.training is False
    assert model.grad_enabled is False
    assert metadata["epoch_zero_based"] == 12
    assert metadata["mean"] == 0.25
    assert metadata["std"] == 1.5


class _AddMixer(torch.nn.Module):
    def __init__(self, amount, *, slots=1):
        super().__init__()
        self.amount = amount
        self.active_slots = slots

    def forward(self, hidden):
        return hidden + self.amount


class _MixerStack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_atom_mixers = torch.nn.ModuleDict(
            {"3": _AddMixer(3), "6": _AddMixer(6), "9": _AddMixer(9)}
        )

    def forward(self, hidden):
        for layer in ("3", "6", "9"):
            hidden = self.neural_atom_mixers[layer](hidden)
        return hidden


def test_scale_one_preserves_mixer_output():
    model = _MixerStack().eval()
    hidden = torch.tensor([2.0])
    expected = model(hidden)

    with inference.scale_slot_return(model, layers=(3, 6, 9), scale=1.0):
        actual = model(hidden)

    torch.testing.assert_close(actual, expected)


def test_scale_zero_removes_only_selected_mixer_and_hooks_restore_on_exception():
    model = _MixerStack().eval()
    hidden = torch.tensor([2.0])
    baseline = model(hidden)

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        with inference.scale_slot_return(model, layers=(6,), scale=0.0):
            intervened = model(hidden)
            torch.testing.assert_close(intervened, hidden + 3 + 9)
            raise RuntimeError("synthetic interruption")

    assert all(not mixer._forward_hooks for mixer in model.neural_atom_mixers.values())
    torch.testing.assert_close(model(hidden), baseline)


def test_train_mode_multislot_and_illegal_layer_are_rejected():
    training_model = _MixerStack().train()
    with pytest.raises(ValueError, match="eval mode"):
        with inference.scale_slot_return(training_model, layers=(6,), scale=0.0):
            pytest.fail("training mode must be rejected before yielding")

    multi_slot_model = _MixerStack().eval()
    multi_slot_model.neural_atom_mixers["6"].active_slots = 2
    with pytest.raises(ValueError, match="single-slot"):
        with inference.scale_slot_return(multi_slot_model, layers=(3, 6, 9), scale=0.0):
            pytest.fail("multi-slot K1 must be rejected before yielding")
    assert all(not mixer._forward_hooks for mixer in multi_slot_model.neural_atom_mixers.values())

    with pytest.raises(ValueError, match="distinct K1 mixer layers"):
        with inference.scale_slot_return(_MixerStack().eval(), layers=(4,), scale=0.0):
            pytest.fail("an unsupported mixer layer must be rejected before yielding")
