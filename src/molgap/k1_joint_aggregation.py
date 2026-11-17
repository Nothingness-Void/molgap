"""Bounded SSMA-style residual on K1's actual incoming gated bond messages.

Construction reuses qm9_neural_atom.make_encoder. The original local sum is
retained for every node. Only incoming degree 1..4 receives the layer-six
residual; no neighbor sampling, attention slots, new bonds or geometry is used.

Mechanism source: https://raw.githubusercontent.com/AlmogDavid/SSMA/master/ssma.py
(accessed 2026-10-01), affine signal and geometric-mean magnitude/summed-phase
stabilization. This compressed stabilized operator has no complete-polynomial
separation guarantee. Phase guarding and bounded logs are explicit extensions.
"""
from __future__ import annotations

import torch
from torch import nn


class BoundedJointAggregation(nn.Module):
    """d64/kappa4 addon; independent_sum is a local capacity control only."""

    def __init__(self, *, mode: str = "ssma", chunk_nodes: int = 256):
        super().__init__()
        if mode not in {"ssma", "independent_sum"}:
            raise ValueError("mode must be ssma or independent_sum")
        if chunk_nodes < 1:
            raise ValueError("chunk_nodes must be positive")
        self.mode = mode
        self.chunk_nodes = chunk_nodes
        self.hidden_channels, self.latent_channels, self.kappa = 192, 64, 4
        self.m1, self.m2 = 5, 253
        self.epsilon = 1e-6
        self.message_norm = nn.LayerNorm(192)
        self.project = nn.Linear(192, 64)
        # Store the official fixed affine operator, counting its storage even
        # though sparse signal construction avoids its dense matrix multiply.
        affine_weight = torch.zeros(1265, 64)
        affine_weight[:64] = -torch.eye(64)
        affine_bias = torch.zeros(1265)
        affine_bias[253] = 1.0
        self.register_buffer("affine_weight", affine_weight)
        self.register_buffer("affine_bias", affine_bias)
        self.compress = nn.Sequential(nn.Linear(1265, 8), nn.Linear(8, 64))
        self.output_norm = nn.LayerNorm(64)
        self.return_projection = nn.Linear(64, 192, bias=False)
        nn.init.zeros_(self.return_projection.weight)

    def stored_numel(self) -> int:
        """Include frozen affine buffers; these are storage, not trainable weights."""
        return sum(value.numel() for value in self.state_dict().values())

    def trainable_numel(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def _signals(self, messages):
        latent = self.project(self.message_norm(messages))
        signal = latent.new_zeros((*latent.shape[:-1], self.m1 * self.m2))
        signal[..., :64] = -latent
        signal[..., 253] = 1.0
        return signal.reshape(*latent.shape[:-1], self.m1, self.m2)

    def _coefficients(self, messages, valid):
        spectrum = torch.fft.fft2(self._signals(messages))
        magnitude = spectrum.abs()
        # atan2 has an unbounded derivative near the complex origin. Masking
        # its input there avoids NaN/large phase gradients even at exact roots.
        safe = torch.where(magnitude >= self.epsilon, spectrum,
                           torch.full_like(spectrum, self.epsilon))
        phase = torch.angle(safe)
        logs = (magnitude + self.epsilon).log().clamp(-30.0, 30.0)
        mask = valid[..., None, None]
        if self.mode == "ssma":
            degree = valid.sum(dim=1).to(logs.dtype)[:, None, None]
            amplitude = (logs.masked_fill(~mask, 0).sum(dim=1) / degree).exp()
            angle = phase.masked_fill(~mask, 0).sum(dim=1)
            mixed = torch.polar(amplitude, angle)
            return torch.fft.ifft2(mixed).real.flatten(start_dim=1)
        # Same affine/FFT/stabilization/compression; each message is transformed
        # independently before addition, with no joint frequency multiplication.
        independent = torch.polar(logs.exp(), phase)
        coefficients = torch.fft.ifft2(independent).real
        return coefficients.masked_fill(~mask, 0).sum(dim=1).flatten(start_dim=1)

    def forward(self, messages, index, dim_size: int):
        if messages.ndim != 2 or messages.shape[1] != 192:
            raise ValueError("Expected gated messages shaped [edges, 192]")
        if index.ndim != 1 or index.numel() != messages.shape[0]:
            raise ValueError("Incoming target index must align with messages")
        if messages.dtype != torch.float32:
            raise ValueError("The frozen bounded SSMA contract requires FP32")
        degree = torch.bincount(index, minlength=dim_size)
        eligible = (degree > 0) & (degree <= self.kappa)
        nodes = eligible.nonzero(as_tuple=False).flatten()
        output = messages.new_zeros((dim_size, 192))
        if nodes.numel() == 0:
            return output
        selected = eligible[index].nonzero(as_tuple=False).flatten()
        order = torch.argsort(index[selected], stable=True)
        selected = selected[order]
        counts = degree[nodes]
        group = torch.repeat_interleave(torch.arange(nodes.numel(), device=index.device), counts)
        starts = torch.cumsum(counts, dim=0) - counts
        slot = torch.arange(selected.numel(), device=index.device) - torch.repeat_interleave(starts, counts)
        packed = messages.new_zeros((nodes.numel(), self.kappa, 192))
        packed[group, slot] = messages[selected]
        valid = torch.arange(self.kappa, device=index.device)[None, :] < counts[:, None]
        updates = []
        for start in range(0, nodes.numel(), self.chunk_nodes):
            end = start + self.chunk_nodes
            coeff = self._coefficients(packed[start:end], valid[start:end])
            updates.append(self.return_projection(self.output_norm(self.compress(coeff))))
        output[nodes] = torch.cat(updates, dim=0)
        return output


def attach_k1_joint_aggregation(model, *, mode="ssma", layer=6, seed=42):
    """Attach hooks without changing original K1 parameter names or RNG state.

    PyG's aggregate hook omits its positional message tensor. A message hook
    supplies the *actual* post-gate tensor, consumed immediately by the aggregate
    hook. This eager hook adapter is not designed for torch.compile, concurrent
    calls to one model, or pickling the whole model; persist state_dict instead.
    """
    if layer != 6:
        raise ValueError("This contract adds a residual only at layer six")
    if hasattr(model, "k1_joint_aggregation"):
        raise ValueError("K1 aggregation addon is already attached")
    from torch_geometric.nn import ResGatedGraphConv

    conv = model.local_blocks[layer - 1].conv
    if not isinstance(conv, ResGatedGraphConv) or conv.out_channels != 192 or conv.aggr != "add":
        raise ValueError("Expected original K1 192-channel additive ResGatedGraphConv")
    with torch.random.fork_rng(devices=[]):
        # torch.manual_seed also seeds CUDA generators; initialize on CPU and
        # seed only its generator so an already allocated backbone is untouched.
        torch.random.default_generator.manual_seed(seed)
        addon = BoundedJointAggregation(mode=mode)
    backbone_parameter = next(model.parameters())
    addon = addon.to(device=backbone_parameter.device, dtype=backbone_parameter.dtype)
    addon.train(model.training)
    model.add_module("k1_joint_aggregation", addon)
    pending = {}

    def capture_messages(module, inputs, output):
        pending["messages"] = output

    def augment_aggregate(module, inputs, output):
        kwargs = inputs[0]
        messages = pending.pop("messages")
        return output + addon(messages, kwargs["index"], output.shape[0])

    model._k1_joint_hook_handles = (
        conv.register_message_forward_hook(capture_messages),
        conv.register_aggregate_forward_hook(augment_aggregate),
    )
    return addon


def backbone_state_dict(model):
    """Return original factory keys, excluding the separately stored addon."""
    return {key: value for key, value in model.state_dict().items()
            if not key.startswith("k1_joint_aggregation.")}


def create_k1_joint_encoder(*, backbone_state=None, mode="ssma", addon_seed=42):
    """Build original K1, load frozen backbone strictly, then attach the addon."""
    from .qm9_neural_atom import make_encoder

    model = make_encoder("neural_atom_k1")
    if backbone_state is not None:
        model.load_state_dict(backbone_state, strict=True)
    attach_k1_joint_aggregation(model, mode=mode, seed=addon_seed)
    return model
