"""Static family hooks for composed pure-2D 500K recipes.

The matched-scale owner retains sampling, evaluation, checkpoints and stages.
These hooks bind accepted initialization and family-specific objectives/EMA.
"""
from pathlib import Path
import json
import math

from .training_reproducibility import sha256_file
from .v4_runtime import state_dict_sha256
from .pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256

K1 = "k1_pretrained_consistency"
K1_MEAN2 = "k1_pretrained_mean2"
K1_ARMS = frozenset({K1, K1_MEAN2})
CONSISTENCY_WEIGHTS = {K1: 0.1, K1_MEAN2: 0.0}
GP = "gptrans_g1_bond_local_ema999"
PARAMETERS = {K1: 3_658_817, K1_MEAN2: 3_658_817, GP: 5_871_201}
INITIAL_FILES = {
    K1: "9656064f7bb0256881da4bc063e90de135171abf517a486bf6202d774694fdba",
    GP: "d471d9f94ff2c10436261b7f70f3e2d9111978f1ac8aadfabea2da32a5fe9a1d",
}
INITIAL_TENSORS = {
    K1: "7458f5c4776208da1715d39cbce66537727ff8054b30da0ea07d82d0e94181f4",
    GP: "a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b",
}
INITIAL_FILES[K1_MEAN2] = INITIAL_FILES[K1]
INITIAL_TENSORS[K1_MEAN2] = INITIAL_TENSORS[K1]
TRANSFORM_SHA256 = "0bbe2b4378afe85fe4fab0a197e0f4b4ec42c5bfb24d58e17576656e42d4a026"


def scientific_contract(arm):
    if arm not in PARAMETERS:
        raise ValueError("Unknown composed scale family")
    return {
        "benchmark_id": "pcqm-composed500k-dev50k-60pass-v1",
        "data_role_fingerprint": FIXED_500K_MANIFEST_SHA256,
        "row_order_fingerprint": "global-randperm-seed42-plus-epoch-drop-last32",
        "feature_fingerprint": "ogb-node9-edge3-rwse16-pure2d",
        "target_fingerprint": "pcqm4mv2-gap-eV", "seed": 42,
        "precision": "fp32", "physical_batch_per_device": 128,
        "device_count": 1, "gradient_accumulation_steps": 1,
        "epochs": 60, "steps_per_epoch": 3906, "sample_exposure": 29_998_080,
        "optimizer_steps": 234_360, "tail_batch_policy": "drop_last",
        "parameters": PARAMETERS[arm], "arm": arm,
        "initial_state_file_sha256": INITIAL_FILES[arm],
        "initial_tensor_sha256": INITIAL_TENSORS[arm],
        "optimizer_fingerprint": "adamw-unfused-foreachFalse-lr4e-4-wd1e-5-clip1" if arm in K1_ARMS else "adamw-unfused-foreachFalse-lr1e-3-wd0.05-clip1",
        "schedule_fingerprint": "cosine60-epoch0-4e-4-epoch59-1e-6" if arm in K1_ARMS else "warmup4-cosine60-min1e-6",
        "loss_fingerprint": ("normalized-gap-two-pass-label-L1-plus0.1-disagreement-MSE" if arm == K1 else "normalized-gap-two-pass-label-L1-plus0-disagreement-MSE") if arm in K1_ARMS else "normalized-gap-L1",
        "target_transform_fingerprint": "all500k-train-only-mean-unbiased-std" if arm in K1_ARMS else TRANSFORM_SHA256,
        "selection_fingerprint": "best-development-live-60epochs" if arm in K1_ARMS else "best-development-EMA0.999-60epochs",
        "ema_decay": None if arm in K1_ARMS else .999,
        "role_access_fingerprint": "official-train-derived-train-and-internal-development-only",
        "geometry_used": False, "teacher_used": False,
        "pretrained_lineage": "retained-100k-stage10-backbone-original-head-reset" if arm in K1_ARMS else "not_applicable",
    }


def make_model(arm, initial_state):
    import torch
    if arm not in PARAMETERS or sha256_file(Path(initial_state)) != INITIAL_FILES[arm]:
        raise ValueError("Composed initialization file identity changed")
    if arm in K1_ARMS:
        from .qm9_neural_atom import make_encoder
        state = torch.load(initial_state, map_location="cpu", weights_only=True)
        if state_dict_sha256(state) != INITIAL_TENSORS[arm]:
            raise ValueError("K1 backbone/head tensor identity changed")
        head = {k: v for k, v in state.items() if k.startswith("head.")}
        if state_dict_sha256(head) != "74109662490a37b16a10f81bd75dd47274465c530ecdb7992061520b65bb0e77":
            raise ValueError("K1 original Gap head was not restored")
        model = make_encoder("neural_atom_k1")
        model.load_state_dict(state, strict=True)
    else:
        from .gptrans_capacity import load_initial
        model = load_initial("degree_bond_local_ema999", Path(initial_state))
        if state_dict_sha256(model.state_dict()) != INITIAL_TENSORS[arm]:
            raise ValueError("G1 local initial tensor identity changed")
    if sum(p.numel() for p in model.parameters()) != PARAMETERS[arm]:
        raise ValueError("Composed architecture parameter count changed")
    return model


def schedule(arm, epoch):
    if type(epoch) is not int or not 0 <= epoch < 60 or arm not in PARAMETERS:
        raise ValueError("Expected a complete-epoch schedule cursor")
    if arm in K1_ARMS:
        return 1e-6 + (4e-4 - 1e-6) * (1 + math.cos(math.pi * epoch / 59)) / 2
    from .pcqm_gptrans_v4 import FrozenEpochScheduler
    return FrozenEpochScheduler.learning_rate(epoch)


def optimizer_for(arm, model):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=schedule(arm, 0),
        weight_decay=1e-5 if arm in K1_ARMS else .05, foreach=False, fused=False)


def make_ema(arm, model):
    if arm not in PARAMETERS:
        raise ValueError("Unknown EMA family")
    if arm in K1_ARMS:
        return None
    from .pcqm_gptrans_v4 import ExponentialMovingAverage
    return ExponentialMovingAverage(model, .999)


def target_statistics(arm, train_graphs, transform_path):
    if arm not in PARAMETERS or len(train_graphs) != 500000:
        raise ValueError("Expected the accepted500K training membership")
    if arm in K1_ARMS:
        from .pcqm_k1_scale_runner import _targets
        values = _targets(train_graphs).double()
        return float(values.mean()), float(values.std(unbiased=True).clamp_min(1e-6))
    if transform_path is None or sha256_file(Path(transform_path)) != TRANSFORM_SHA256:
        raise ValueError("G1 target transform differs from accepted100K asset")
    transform = json.loads(Path(transform_path).read_bytes())
    return float(transform["mean"]), float(transform["std"])


def strip_geometry(roles):
    from .k1_screen_training import FORBIDDEN_MODEL_FIELDS
    for graphs in roles.values():
        for dataset in graphs.datasets:
            for field in FORBIDDEN_MODEL_FIELDS:
                if field in dataset._data:
                    del dataset._data[field]
                    dataset.slices.pop(field, None)


def _objective(arm, model, batch, mean, std):
    import torch.nn.functional as F
    from .pcqm_gptrans_v4 import _forward
    target = (batch.y.view(-1) - mean) / std
    first = _forward(model, batch)
    if arm in K1_ARMS:
        from .k1_pretrained_combo import objective
        second = _forward(model, batch)
        loss, terms = objective(first, second, target, mode="pretrained_consistency",
                                consistency_weight=CONSISTENCY_WEIGHTS[arm])
        return loss, terms
    if arm != GP:
        raise ValueError("Unknown objective family")
    loss = F.l1_loss(first, target)
    return loss, {"supervised_l1": loss, "combined_loss": loss}


def step(arm, model, optimizer, batch, mean, std):
    import torch
    if batch.num_graphs != 128:
        raise ValueError("Expected physicalBS128")
    optimizer.zero_grad(set_to_none=True)
    loss, terms = _objective(arm, model, batch, mean, std)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    step.last_components = {k: v.detach() for k, v in terms.items() if v is not None}
    return terms["supervised_l1"].detach()


def qualify_signal(arm, model, batch, mean, std):
    import torch
    model.train()
    if arm in K1_ARMS:
        from .pcqm_gptrans_v4 import _forward
        first, second = _forward(model, batch), _forward(model, batch)
        disagreement = (first - second).square().mean()
        gradients = torch.autograd.grad(disagreement, tuple(model.parameters()), allow_unused=True)
        norm = sum(float(g.detach().square().sum()) for g in gradients if g is not None)
        value = float(disagreement.detach())
        if not math.isfinite(value) or not value > 0 or not math.isfinite(norm) or not norm > 0:
            raise ValueError("Missing nonzero finite dropout disagreement/gradient")
        return {"accepted": True, "disagreement": value, "gradient_squared_norm": norm, "teacher_used": False}
    if arm != GP:
        raise ValueError("Unknown qualification family")
    from .gptrans_capacity import BondLocalBlock
    blocks = [b for b in model.blocks if isinstance(b, BondLocalBlock)]
    if len(blocks) != 12 or any(bool(b.output.weight.detach().count_nonzero()) or bool(b.output.bias.detach().count_nonzero()) for b in blocks):
        raise ValueError("Real-bond stream is not an identity-initialized twelve-block addon")
    return {"accepted": True, "local_blocks": len(blocks), "output_projection_initially_zero": True, "ema_decay": .999}
