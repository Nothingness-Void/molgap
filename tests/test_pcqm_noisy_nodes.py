import torch
from molgap.noisy_nodes import GPTransNoisyNodes, NoisyNodesConfig
from molgap.gptrans import OGBGPTransTiny
from molgap.training_reproducibility import configure_fp32_determinism


def make_dummy_batch():
    x = torch.zeros((6, 9), dtype=torch.long)
    x[:, 0] = torch.tensor([6, 6, 8, 7, 6, 1], dtype=torch.long)  # C, C, O, N, C, H
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]], dtype=torch.long
    )
    edge_attr = torch.zeros((edge_index.shape[1], 3), dtype=torch.long)
    batch = type("Batch", (), {})()
    batch.x = x
    batch.edge_index = edge_index
    batch.edge_attr = edge_attr
    batch.batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    return batch


def test_noisy_nodes_parameters():
    model = GPTransNoisyNodes(noise_std=0.15, loss_weight=0.1)
    base = OGBGPTransTiny()
    base_params = sum(p.numel() for p in base.parameters())
    noisy_params = sum(p.numel() for p in model.parameters())
    aux_params = sum(p.numel() for p in model.denoise_head.parameters())
    assert aux_params == 256 * 119 + 119
    assert noisy_params == base_params + aux_params


def test_frozen_core_load_preserves_only_auxiliary_head(tmp_path):
    from molgap.training_reproducibility import sha256_file
    from molgap.v4_runtime import load_frozen_initial_state, model_state_sha256

    configure_fp32_determinism(42)
    core = OGBGPTransTiny()
    state = core.state_dict()
    artifact = tmp_path / "initial_state.pt"
    torch.save({"format": "test-frozen-state", "model_state": state,
                "state_sha256": model_state_sha256(core)}, artifact)
    candidate = GPTransNoisyNodes()
    auxiliary_before = {key: value.clone() for key, value in candidate.state_dict().items()
                        if key.startswith("denoise_head.")}
    load_frozen_initial_state(
        candidate, artifact,
        expected_file_sha256=sha256_file(artifact),
        expected_state_sha256=model_state_sha256(core),
        expected_format="test-frozen-state",
        allowed_missing_keys=("denoise_head.weight", "denoise_head.bias"),
    )
    for key, value in state.items():
        assert torch.equal(candidate.state_dict()[key], value)
    for key, value in auxiliary_before.items():
        assert torch.equal(candidate.state_dict()[key], value)


def test_noisy_nodes_forward_train_and_backward():
    configure_fp32_determinism(42)
    model = GPTransNoisyNodes(noise_std=0.15, loss_weight=0.1)
    model.train()
    batch = make_dummy_batch()
    pred, aux_loss = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        return_aux_loss=True,
    )
    assert pred.shape == (2, 1)
    assert torch.isfinite(pred).all()
    assert aux_loss.ndim == 0
    assert torch.isfinite(aux_loss)
    assert aux_loss > 0.0

    total_loss = pred.abs().mean() + 0.1 * aux_loss
    total_loss.backward()

    # Verify gradients are finite for both backbone and denoise head
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing grad for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


def test_noisy_nodes_eval_mode_determinism():
    configure_fp32_determinism(42)
    model = GPTransNoisyNodes(noise_std=0.15, loss_weight=0.1)
    model.eval()
    batch = make_dummy_batch()

    with torch.no_grad():
        out1 = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        out2 = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    assert torch.equal(out1, out2), "Eval mode must be strictly deterministic (zero noise)"


def test_noisy_nodes_optimizer_step():
    from molgap.noisy_nodes import _optimizer_step_noisy_nodes
    from molgap.pcqm_gptrans_v4 import ExponentialMovingAverage, make_adamw_compat

    configure_fp32_determinism(42)
    model = GPTransNoisyNodes(noise_std=0.15, loss_weight=0.1)
    optimizer = make_adamw_compat(
        model.parameters(), lr=1e-3, weight_decay=1e-2, fused=False, foreach=False
    )
    ema = ExponentialMovingAverage(model)
    batch = make_dummy_batch()
    batch.y = torch.tensor([4.2, 5.1], dtype=torch.float)
    mean = torch.tensor(4.5)
    std = torch.tensor(1.2)
    gap_loss, aux_loss = _optimizer_step_noisy_nodes(
        model, optimizer, ema, batch, mean, std, check_finite=True
    )
    assert gap_loss > 0.0
    assert aux_loss > 0.0
