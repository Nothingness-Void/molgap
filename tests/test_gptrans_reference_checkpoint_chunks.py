"""CPU-only checks for the execution-only GPTrans checkpoint chunks."""
import pytest
import torch

from molgap import pcqm_gptrans_v4 as owner


class _EMA:
    decay = owner.EMA_DECAY

    def __init__(self, model):
        self._state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def state_dict(self):
        return self._state


def _checkpoint_kwargs(tmp_path, *, execution_only_chunking=False):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    return dict(
        path=tmp_path / "last_checkpoint.pt",
        epoch=9,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=_EMA(model),
        trace=[],
        best=1.0,
        best_epoch=9,
        target_stats={"mean_eV": 5.0, "sample_std_eV": 1.0},
        runtime_certificate_id="certificate",
        source_archive_sha256="a" * 64,
        execution_only_chunking=execution_only_chunking,
    )


def test_execution_only_chunk_is_atomic_and_independently_readable(tmp_path):
    checkpoint = tmp_path / "last_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-v1")

    chunk = owner._publish_checkpoint_chunk(tmp_path, checkpoint, 9)

    assert chunk.name == "checkpoint_epoch_09.pt"
    assert chunk.read_bytes() == checkpoint.read_bytes()
    assert not (tmp_path / "checkpoint_epoch_09.pt.tmp").exists()

    checkpoint.write_bytes(b"checkpoint-v2")
    replaced = owner._publish_checkpoint_chunk(tmp_path, checkpoint, 9)
    assert replaced == chunk
    assert chunk.read_bytes() == b"checkpoint-v2"


def test_checkpoint_payload_records_execution_only_mode_without_v5_audit(tmp_path):
    payload = owner._save_checkpoint(**_checkpoint_kwargs(tmp_path, execution_only_chunking=True))

    assert payload["execution_only_chunking"] is True
    assert "v5_audit" not in payload
    assert torch.load(tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False)[
        "execution_only_chunking"] is True

    plain = owner._save_checkpoint(**_checkpoint_kwargs(tmp_path / "plain"))
    assert "execution_only_chunking" not in plain


def test_execution_only_mode_rejects_v5_audit_before_gpu_setup(tmp_path):
    with pytest.raises(ValueError, match="cannot enable V5 audit"):
        owner.run_training(
            dataset_root=tmp_path,
            manifest_path=tmp_path / "manifest.json",
            preflight_path=tmp_path / "preflight.json",
            source_archive=tmp_path / "source.tar.gz",
            source_archive_sha256="a" * 64,
            source_commit="b" * 40,
            output=tmp_path / "output",
            platform_id="kaggle",
            initial_state_path=tmp_path / "initial_state.pt",
            v5_audit=True,
            execution_only_chunking=True,
        )


def test_execution_only_flag_keeps_owner_abi_opt_in():
    import inspect

    parameter = inspect.signature(owner.run_training).parameters["execution_only_chunking"]
    assert parameter.default is False
