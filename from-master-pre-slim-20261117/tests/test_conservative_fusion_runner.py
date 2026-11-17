from __future__ import annotations

import numpy as np
import torch

from molgap.conservative_fusion_payload import EXTERNAL_FORMAT, TRAINING_FORMAT
from molgap.conservative_fusion_runner import run_conservative_fusion
from molgap.hierarchical_fusion import ConservativeFusionConfig


def test_conservative_runner_writes_durable_rejected_identity(tmp_path) -> None:
    rng = np.random.default_rng(31)
    rows = 80
    targets = rng.normal(size=(rows, 3)).astype(np.float32)
    context = rng.normal(size=(rows, 9)).astype(np.float32)
    training = {
        "format": TRAINING_FORMAT,
        "source_idx": torch.arange(rows),
        "targets": torch.from_numpy(targets),
        "equal_prediction": torch.from_numpy(targets.copy()),
        "dense_prediction": torch.from_numpy(targets.copy()),
        "context": torch.from_numpy(context).half(),
        "train_indices": torch.arange(0, 50),
        "validation_indices": torch.arange(50, 65),
        "test_indices": torch.arange(65, 80),
    }
    external_rows = 20
    external_targets = rng.normal(size=(external_rows, 3)).astype(np.float32)
    external = {
        "format": EXTERNAL_FORMAT,
        "source_idx": torch.arange(external_rows),
        "targets": torch.from_numpy(external_targets),
        "equal_prediction": torch.from_numpy(external_targets.copy()),
        "dense_prediction": torch.from_numpy(external_targets.copy()),
        "routed_v4_prediction": torch.from_numpy(external_targets.copy()),
        "context": torch.from_numpy(
            rng.normal(size=(external_rows, 9)).astype(np.float32)
        ).half(),
        "scope": torch.tensor([1] * 10 + [2] * 10, dtype=torch.int8),
    }
    training_path = tmp_path / "training.pt"
    external_path = tmp_path / "external.pt"
    torch.save(training, training_path)
    torch.save(external, external_path)
    result = run_conservative_fusion(
        training_payload_path=training_path,
        external_payload_path=external_path,
        checkpoint_dir=tmp_path / "checkpoints",
        results_dir=tmp_path / "results",
        device="cpu",
        seeds=(42,),
        config=ConservativeFusionConfig(
            hidden_channels=8,
            epochs=2,
            patience=1,
            batch_size=32,
        ),
    )
    assert result["status"] == "rejected"
    assert result["promotion_gate"]["equal"]["passed"] is False
    assert (tmp_path / "results" / "completion_manifest.json").is_file()
    assert (tmp_path / "checkpoints" / "equal_seed42.best.pt").is_file()
    assert (tmp_path / "checkpoints" / "dense_seed42.best.pt").is_file()
