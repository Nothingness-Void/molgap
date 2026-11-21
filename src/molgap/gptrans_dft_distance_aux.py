"""GPT-T pair-update-norm plus a train-only DFT heavy-atom distance head.

Contract: 3D-supervised training, pure-2D inference. The official PCQM4Mv2 DFT
coordinates (a train-row sidecar) supply an auxiliary label only. The head reads
the final dense pair state, is never called by the evaluation forward, and no
coordinate is needed to build, load or score a checkpoint.

Frozen constants: loss weight 0.1 on the normalized Gap L1 scale, L1 in Angstrom
over all real heavy-atom pairs i<j of rows with valid coordinates, head
LayerNorm -> Linear -> GELU -> Linear on pair_channels, symmetrized.
"""
import json
from copy import copy
from dataclasses import replace
from pathlib import Path

from .gptrans_bond_pair_norm import construct_family_model

MODE = "degree_bond_local_pair_update_norm_dft_distance_ema999"
ADDON_NAME = "dft_distance"
LOSS_WEIGHT = 0.1
EXTRA_FIELDS = ("dft_pos", "dft_valid")
SIDECAR_FORMAT = "molgap-pcqm-dft-geometry-sidecar-v1"
SHARD_FORMAT = "molgap-pcqm-dft-geometry-shard-v1"


def _build_addon_class():
    import torch
    from torch import nn

    class DistanceHeadAddon(nn.Module):
        """Identity on the readout slot (so it is a legal substrate attachment) that owns the head."""

        def __init__(self, channels, seed):
            super().__init__()
            # nn.Linear draws from the global RNG; restore it so data order and dropout match the control.
            with torch.random.fork_rng(devices=[]):
                self.head = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels),
                                          nn.GELU(), nn.Linear(channels, 1))
            generator = torch.Generator().manual_seed(int(seed))
            with torch.no_grad():
                for layer in (self.head[1], self.head[3]):
                    bound = layer.in_features ** -0.5
                    layer.weight.uniform_(-bound, bound, generator=generator)
                    layer.bias.uniform_(-bound, bound, generator=generator)
            self.capture, self.pair = False, None
            self.last_aux = None

        def forward(self, payload):
            return payload  # evaluation and Gap forward never touch the head

    return DistanceHeadAddon


def mount(recipe, model):
    """After the frozen init load, before optimizer/EMA: attach the head and the capture hook."""
    from .models.substrate import ModelSubstrate, SlotBinding, SlotName
    addon = _build_addon_class()(model.pair_channels, recipe.seed)
    ModelSubstrate(model, "gptrans_t").attach(ADDON_NAME, addon, SlotBinding("gptrans_t", SlotName.PRE_READOUT))

    def grab(module, args, output):
        if addon.capture:
            addon.pair = output[1]
    model.blocks[-1].register_forward_hook(grab)
    return model


def objective(model, batch, forward, recipe):
    """(loss, (absolute, rows)) with the reference Gap L1 plus the weighted distance term."""
    import torch
    import torch.nn.functional as functional
    addon = getattr(model, "_molgap_addons")[ADDON_NAME]
    addon.capture, addon.pair = True, None
    try:
        prediction = forward(model, batch)
    finally:
        addon.capture = False
    pair, addon.pair = addon.pair, None
    target = (batch.y.view(-1).float() - recipe.target_mean_eV) / recipe.target_std_eV
    gap = functional.l1_loss(prediction, target)
    rows = int(batch.num_graphs)
    metrics = (gap.detach() * rows, rows)
    layout = batch.execution_layout
    size, widest = layout["batch_size"], layout["max_nodes"]
    valid = batch.dft_valid.view(-1).bool()
    real = layout["mask"] & valid[:, None]
    pairs = real[:, :, None] & real[:, None, :] & torch.triu(
        torch.ones(widest, widest, dtype=torch.bool, device=real.device), diagonal=1)
    graph, first, second = pairs.nonzero(as_tuple=True)
    if graph.numel() == 0:
        addon.last_aux = gap.new_zeros(())
        return gap, metrics
    coordinates = batch.dft_pos.new_zeros(size * widest, 3)
    coordinates[layout["flat_index"]] = batch.dft_pos
    coordinates = coordinates.view(size, widest, 3)
    distance = (coordinates[graph, first] - coordinates[graph, second]).norm(dim=-1)
    # Local index 0 of the dense pair state is the graph token; atom j sits at j + 1.
    states = torch.cat((pair[graph, :, first + 1, second + 1], pair[graph, :, second + 1, first + 1]))
    forward_half, reverse_half = addon.head(states).squeeze(-1).chunk(2)
    aux = (0.5 * (forward_half + reverse_half) - distance).abs().mean()
    addon.last_aux = aux.detach()
    return gap + LOSS_WEIGHT * aux, metrics


class _DftTrainDataset:
    """Train rows plus node-level DFT positions; development is never wrapped."""
    transform = None

    def __init__(self, base, positions, offsets, valid):
        self.base, self.positions, self.offsets, self.valid = base, positions, offsets, valid

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        import torch
        graph = copy(self.base[index])
        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        graph.dft_pos = self.positions[start:stop]
        graph.dft_valid = self.valid[index:index + 1]
        if graph.dft_pos.shape[0] != graph.x.shape[0]:
            raise ValueError("DFT sidecar atom count differs from the training graph")
        return graph


def _train_atom_counts(train):
    import torch
    parts = getattr(train, "datasets", (train,))
    counts = []
    for part in parts:
        slices = part.slices["x"]
        counts.append(slices[1:] - slices[:-1])
    return torch.cat(counts).to(torch.int64)


def attach_inputs(input_root, sidecar, data):
    """Verify the pinned sidecar fail-closed and attach coordinates to train rows only."""
    import torch
    from .family_training_data import DatasetSelection, resolve_pinned_manifest
    from .experiment_family_workflow import _artifact_path
    from .training_reproducibility import sha256_file
    if type(sidecar) is not DatasetSelection:
        raise ValueError("DFT distance mode requires a pinned input sidecar")
    path = resolve_pinned_manifest(Path(input_root), sidecar)  # also checks the manifest sha256
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != SIDECAR_FORMAT or manifest.get("status") != "complete":
        raise ValueError("DFT sidecar manifest is not a complete v1 sidecar")
    rows = data.train_rows
    span = manifest.get("rows", {})
    if data.train_source_start != 0 or span.get("source_idx_start") != 0 or span.get("source_idx_stop", -1) < rows:
        raise ValueError("DFT sidecar does not cover the train role rows 0..train_rows-1")
    counts = _train_atom_counts(data.train)
    if counts.numel() != rows:
        raise ValueError("Train atom counts differ from the train row count")
    positions, valid, sidecar_counts, cursor = [], [], [], 0
    for shard in manifest["shards"]:
        start, stop = shard["source_idx_start"], shard["source_idx_stop"]
        if cursor >= rows:
            break
        if start != cursor or stop - start != shard["rows"]:
            raise ValueError("DFT sidecar shards are not contiguous from row 0")
        file = _artifact_path(path.parent, shard["file"])
        if file.stat().st_size != shard["bytes"] or sha256_file(file) != shard["sha256"]:
            raise ValueError("DFT sidecar shard bytes differ from the manifest")
        content = torch.load(file, map_location="cpu", weights_only=True)
        offsets = content["atom_offsets"]
        if (content["format"] != SHARD_FORMAT
                or not torch.equal(content["source_idx"], torch.arange(start, stop))
                or offsets.shape != (shard["rows"] + 1,) or int(offsets[0]) != 0
                or content["pos"].shape != (int(offsets[-1]), 3) or content["valid"].shape != (shard["rows"],)
                or content["valid"].dtype != torch.bool or content["pos"].dtype != torch.float32):
            raise ValueError("DFT sidecar shard content is malformed")
        keep = min(stop, rows) - start
        positions.append(content["pos"][:int(offsets[keep])])
        valid.append(content["valid"][:keep])
        sidecar_counts.append((offsets[1:keep + 1] - offsets[:keep]))
        cursor = start + keep
    if cursor != rows:
        raise ValueError("DFT sidecar is missing train rows")
    positions, valid = torch.cat(positions), torch.cat(valid)
    sidecar_counts = torch.cat(sidecar_counts)
    if not bool((sidecar_counts[valid] == counts[valid]).all()):
        raise ValueError("DFT sidecar atom counts differ from the training graphs")
    # Rows without valid coordinates must carry no atoms or the exact graph atom count.
    if not bool(((sidecar_counts == counts) | (~valid & (sidecar_counts == 0))).all()):
        raise ValueError("DFT sidecar invalid-row atom counts are inconsistent")
    if not bool(torch.isfinite(positions).all()):
        raise ValueError("DFT sidecar positions are not finite")
    # Dense per-graph offsets; invalid zero-atom rows get zero padding of the graph's atom count.
    full_offsets = torch.zeros(rows + 1, dtype=torch.int64)
    full_offsets[1:] = torch.cumsum(counts, 0)
    if int(full_offsets[-1]) != positions.shape[0]:
        padded = positions.new_zeros(int(full_offsets[-1]), 3)
        source = torch.cumsum(sidecar_counts, 0) - sidecar_counts
        for row in torch.nonzero(sidecar_counts > 0).view(-1).tolist():
            n = int(counts[row])
            padded[int(full_offsets[row]):int(full_offsets[row]) + n] = positions[int(source[row]):int(source[row]) + n]
        positions = padded
    return replace(data, train=_DftTrainDataset(data.train, positions, full_offsets, valid))


def policy(mode):
    from .family_training_policy import FamilyModePolicy
    if mode != MODE:
        raise ValueError("Unknown DFT distance auxiliary mode")
    # objective_name stays None: the recipe mode names the (different) objective identity.
    return FamilyModePolicy(construct=construct_family_model, mount=mount, objective=objective,
                            attach_inputs=attach_inputs, input_sidecar_required=True,
                            extra_batch_fields=EXTRA_FIELDS)
