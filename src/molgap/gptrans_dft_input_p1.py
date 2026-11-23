"""GPT-T with DFT distances as a training-time input to the initial pair state (P1).

Contract: 3D-supervised training, pure-2D inference. For a share of training samples the
official DFT heavy-atom coordinates (the C1 train-row sidecar) enter the model as RBF
distances added to the initial pair state. Evaluation and inference set no distance state,
so every hook returns its input unchanged and the forward pass is the control's.

Two modes share this module. P1 adds the denoising head (loss on clean distances); zeros
uses the same structure, masking and mode schedule with all-zero RBF features and no loss.
Constants are frozen in experiments/pcqm_gptrans_dft_input_p1_20261123/protocol.md.
"""
import hashlib
from types import SimpleNamespace

from .gptrans_bond_pair_norm import construct_family_model
from .gptrans_dft_distance_aux import EXTRA_FIELDS, _build_addon_class as _distance_head_class, attach_inputs

MODE_P1 = "degree_bond_local_pair_update_norm_dft_input_p1_ema999"
MODE_ZEROS = "degree_bond_local_pair_update_norm_dft_input_zeros_ema999"
ADDON_NAME = "dft_input"
KERNELS, RBF_MAX_A, NOISE_A, LOSS_WEIGHT = 64, 10.0, 0.2, 0.1
PROBS = (0.40, 0.20)  # P(2D), P(3D-only); the rest (0.40) is 2D+3D. Tests may patch this tuple.


def _build_addon_class():
    import torch
    from torch import nn

    head_class = _distance_head_class()

    class DistanceInputAddon(nn.Module):
        """Owns the distance MLP (and, for P1, the denoising head); the slot call is an identity."""

        def __init__(self, channels, seed, denoise):
            super().__init__()
            # Own generator and a restored global RNG: data order and dropout match the control.
            with torch.random.fork_rng(devices=[]):
                self.mlp = nn.Sequential(nn.Linear(KERNELS, channels), nn.GELU(), nn.Linear(channels, channels))
                self.head = head_class(channels, seed) if denoise else None
            generator = torch.Generator().manual_seed(int(seed) + 7919)
            with torch.no_grad():
                first = self.mlp[0]
                bound = first.in_features ** -0.5
                first.weight.uniform_(-bound, bound, generator=generator)
                first.bias.uniform_(-bound, bound, generator=generator)
                nn.init.zeros_(self.mlp[2].weight)  # the branch adds exactly zero at step 0
                nn.init.zeros_(self.mlp[2].bias)
            self.register_buffer("centers", torch.linspace(0.0, RBF_MAX_A, KERNELS), persistent=False)
            self.sigma = RBF_MAX_A / (KERNELS - 1)
            self.state = None  # set by the objective (training) or score_mode (offline), never by evaluation

        def forward(self, payload):
            return payload

    return DistanceInputAddon


def mount(recipe, model):
    """After the frozen init load, before optimizer/EMA: attach the addon and the model hooks."""
    import torch.nn.functional as functional
    from .models.substrate import ModelSubstrate, SlotBinding, SlotName
    addon = _build_addon_class()(model.pair_channels, recipe.seed, denoise=recipe.mode == MODE_P1)
    ModelSubstrate(model, "gptrans_t").attach(ADDON_NAME, addon, SlotBinding("gptrans_t", SlotName.PRE_READOUT))

    def mask_spatial(module, args, output):  # (B, N, N, C): 3D-only samples lose the SPD encoding
        state = addon.state
        if state is not None and state.masked:
            return output.masked_fill(state.drop_pair[:, None, None, None], 0.0)

    def mask_bond(module, args, output):  # (E, C): ... and the bond encoding
        state = addon.state
        if state is not None and state.masked:
            return output.masked_fill(state.drop_pair[state.edge_graph][:, None], 0.0)

    def add_distance(module, args, kwargs):  # pre-hook of block 0: the initial pair state
        state = addon.state
        if state is None:
            return None
        pair = args[1] + functional.pad(state.delta, (1, 0, 1, 0))  # graph-token row and column get none
        return (args[0], pair, *args[2:]), kwargs

    def grab(module, args, output):  # final pair state for the denoising head
        if addon.state is not None and addon.state.capture:
            addon.state.final_pair = output[1]
    model.spatial_encoder.register_forward_hook(mask_spatial)
    model.bond_encoder.register_forward_hook(mask_bond)
    model.blocks[0].register_forward_pre_hook(add_distance, with_kwargs=True)
    model.blocks[-1].register_forward_hook(grab)
    return model


def _distances(batch, positions):
    """Dense heavy-atom distances (B, N, N) of a padded batch, plus its real-atom mask."""
    import torch
    layout = batch.execution_layout
    size, widest = layout["batch_size"], layout["max_nodes"]
    coordinates = positions.new_zeros(size * widest, 3)
    coordinates[layout["flat_index"]] = positions
    coordinates = coordinates.view(size, widest, 3)
    return (coordinates[:, :, None] - coordinates[:, None, :]).pow(2).sum(-1).sqrt(), layout["mask"]


def set_inputs(model, batch, use3d, only3d, noise=None, *, capture=False):
    """Build the per-batch distance state. use3d/only3d: bool (B,) on the batch device."""
    import torch
    addon = getattr(model, "_molgap_addons")[ADDON_NAME]
    positions = batch.dft_pos if noise is None else batch.dft_pos + noise
    distance, real = _distances(batch, positions)
    if addon.head is None:  # zeros arm: the MLP sees an all-zero feature vector
        features = distance.new_zeros(*distance.shape, KERNELS)
    else:
        features = torch.exp(-0.5 * ((distance[..., None] - addon.centers) / addon.sigma).pow(2))
    delta = addon.mlp(features).permute(0, 3, 1, 2)
    pairs = real[:, :, None] & real[:, None, :] & use3d[:, None, None]
    delta = torch.where(pairs[:, None], delta, 0.0)
    addon.state = SimpleNamespace(delta=delta, drop_pair=only3d, masked=True, edge_graph=batch.execution_layout["edges"][0],
                                  capture=capture, final_pair=None)
    return addon, pairs


def clear_inputs(model):
    getattr(model, "_molgap_addons")[ADDON_NAME].state = None


def batch_generator(seed, batch):
    """One CPU generator per batch from (recipe seed, row identities): no global RNG, no checkpoint state."""
    import torch
    rows = batch.source_idx.view(-1).long().cpu().numpy().tobytes()
    digest = hashlib.sha256(int(seed).to_bytes(8, "little") + rows).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") >> 1)


def draw_modes(seed, batch):
    """(use3d, only3d, noise) on the batch device. Modes first, then noise, so P1 and zeros match."""
    import torch
    generator = batch_generator(seed, batch)
    u = torch.rand(batch.num_graphs, generator=generator)
    noise = torch.randn(batch.dft_pos.shape[0], 3, generator=generator) * NOISE_A
    device = batch.dft_pos.device
    u = u.to(device)
    valid = batch.dft_valid.view(-1).bool()  # a row without geometry is always 2D; the draw was made first
    use3d = (u >= PROBS[0]) & valid
    only3d = use3d & (u < PROBS[0] + PROBS[1])
    return use3d, only3d, noise.to(device)


def objective(model, batch, forward, recipe):
    """(loss, (absolute, rows)): the reference Gap L1, plus 0.1 x clean-distance L1 for P1."""
    import torch
    import torch.nn.functional as functional
    use3d, only3d, noise = draw_modes(recipe.seed, batch)
    addon, pairs = set_inputs(model, batch, use3d, only3d, noise, capture=recipe.mode == MODE_P1)
    try:
        prediction = forward(model, batch)
        final_pair = addon.state.final_pair
    finally:
        addon.state = None
    target = (batch.y.view(-1).float() - recipe.target_mean_eV) / recipe.target_std_eV
    gap = functional.l1_loss(prediction, target)
    rows = int(batch.num_graphs)
    metrics = (gap.detach() * rows, rows)
    if addon.head is None:
        return gap, metrics
    clean, _ = _distances(batch, batch.dft_pos)
    widest = batch.execution_layout["max_nodes"]
    pairs = pairs & torch.triu(torch.ones(widest, widest, dtype=torch.bool, device=pairs.device), diagonal=1)
    states = final_pair[:, :, 1:widest + 1, 1:widest + 1].permute(0, 2, 3, 1)
    predicted = addon.head.head(states).squeeze(-1)
    predicted = 0.5 * (predicted + predicted.transpose(1, 2))
    aux = torch.where(pairs, (predicted - clean).abs(), 0.0).sum() / pairs.sum().clamp_min(1)
    addon.last_aux = aux.detach()
    return gap + LOSS_WEIGHT * aux, metrics


def policy(mode):
    from .family_training_policy import FamilyModePolicy
    if mode not in (MODE_P1, MODE_ZEROS):
        raise ValueError("Unknown DFT input mode")
    # objective_name stays None: the recipe mode names the (different) objective identity.
    return FamilyModePolicy(construct=construct_family_model, mount=mount, objective=objective,
                            attach_inputs=attach_inputs, input_sidecar_required=True,
                            extra_batch_fields=EXTRA_FIELDS)
