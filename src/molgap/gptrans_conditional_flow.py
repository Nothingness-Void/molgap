"""Zero-initialized conditional allocation for the frozen GPTrans flow paths."""
from __future__ import annotations

import torch
import torch.nn as nn

from .gptrans import GPTransBlock, GraphPropagationAttention


MODES = ("conditional_pair_readback", "conditional_pair_recurrence")


class ConditionalPairReadbackAttention(GraphPropagationAttention):
    """Condition the necessary pair-to-node message without deleting it."""

    def __init__(self, node_channels, pair_channels, num_heads, dropout):
        super().__init__(node_channels, pair_channels, num_heads, dropout)
        self.conditional_readback = nn.Linear(node_channels + pair_channels, 1)
        nn.init.zeros_(self.conditional_readback.weight)
        nn.init.zeros_(self.conditional_readback.bias)

    def forward(self, node, pair, key_padding_mask):
        batch_size, node_count, channels = node.shape
        qkv = self.qkv(node).reshape(
            batch_size, node_count, 3, self.num_heads, self.head_channels
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        logits = (query @ key.transpose(-2, -1)) * self.scale
        logits = logits + self.pair_to_attention(pair)
        residual_logits = logits
        logits = logits.masked_fill(key_padding_mask, float("-inf"))
        attention = self.attention_dropout(torch.softmax(logits, dim=-1))
        node_update = (attention @ value).transpose(1, 2).reshape(
            batch_size, node_count, channels
        )

        pair_update = self.attention_to_pair(attention + residual_logits)
        pair_weights = torch.softmax(
            pair_update.masked_fill(key_padding_mask, float("-inf")), dim=-1
        )
        pair_message = (pair_weights * pair_update).sum(-1).transpose(1, 2)
        gate_input = torch.cat((node, pair_message), dim=-1)
        gate = 1.0 + torch.tanh(self.conditional_readback(gate_input))
        node_update = node_update + self.pair_to_node(gate * pair_message)
        return self.output_dropout(self.output(node_update)), pair_update


class ConditionalPairRecurrenceBlock(GPTransBlock):
    """Condition recurrent pair accumulation while preserving direct readback."""

    def __init__(
        self,
        node_channels,
        pair_channels,
        num_heads,
        dropout,
        drop_path,
        layer_scale,
    ):
        super().__init__(
            node_channels,
            pair_channels,
            num_heads,
            dropout,
            drop_path,
            layer_scale,
        )
        self.conditional_recurrence = nn.Conv2d(pair_channels, pair_channels, 1)
        nn.init.zeros_(self.conditional_recurrence.weight)
        nn.init.zeros_(self.conditional_recurrence.bias)

    def forward(self, node, pair, key_padding_mask):
        node_update, pair_update = self.attention(
            self.node_norm1(node), pair, key_padding_mask
        )
        gate = 1.0 + torch.tanh(self.conditional_recurrence(pair))
        pair = pair + gate * pair_update
        node = node + self.drop_path(self.attention_scale * node_update)
        node = node + self.drop_path(self.ffn_scale * self.ffn(self.node_norm2(node)))
        return node, pair


def _load_preserved_state(replacement, original, added_prefix: str) -> None:
    missing, unexpected = replacement.load_state_dict(
        original.state_dict(), strict=False
    )
    if unexpected or not missing or any(added_prefix not in name for name in missing):
        raise RuntimeError(
            f"Conditional replacement state mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )


def apply_conditional_flow(model, variant: str):
    if variant not in MODES:
        raise ValueError(variant)
    with torch.random.fork_rng(devices=[]):
        if variant == "conditional_pair_readback":
            for block in model.blocks:
                original = block.attention
                replacement = ConditionalPairReadbackAttention(
                    original.qkv.in_features,
                    original.pair_to_attention.in_channels,
                    original.num_heads,
                    original.attention_dropout.p,
                )
                _load_preserved_state(
                    replacement, original, "conditional_readback"
                )
                block.attention = replacement
        else:
            for index, original in enumerate(model.blocks):
                channels = original.node_norm1.normalized_shape[0]
                replacement = ConditionalPairRecurrenceBlock(
                    channels,
                    original.attention.pair_to_attention.in_channels,
                    original.attention.num_heads,
                    original.attention.attention_dropout.p,
                    original.drop_path.probability,
                    1.0,
                )
                _load_preserved_state(
                    replacement, original, "conditional_recurrence"
                )
                model.blocks[index] = replacement
    return model


def check_conditional_mechanism(variant: str, device: str = "cuda") -> dict:
    if variant not in MODES:
        raise ValueError(variant)
    torch.manual_seed(42)
    mask = torch.tensor([[[[False, False, False, True]]]], device=device)
    node = torch.randn(1, 4, 256, device=device)
    pair = torch.randn(1, 32, 4, 4, device=device)

    if variant == "conditional_pair_readback":
        reference = GraphPropagationAttention(256, 32, 8, 0.0).to(device).eval()
        candidate = ConditionalPairReadbackAttention(256, 32, 8, 0.0).to(device).eval()
        _load_preserved_state(candidate, reference, "conditional_readback")
        reference_node, reference_pair = reference(node, pair, mask)
        candidate_node, candidate_pair = candidate(node, pair, mask)
        if not torch.equal(reference_node, candidate_node) or not torch.equal(
            reference_pair, candidate_pair
        ):
            raise RuntimeError("Readback gate is not reference-equivalent at initialization")
        with torch.no_grad():
            candidate.conditional_readback.weight.fill_(0.01)
        changed_node, _ = candidate(node, pair, mask)
        return {
            "initial_output_bitwise_equal": True,
            "preserved_pair_recurrence": True,
            "conditional_readback_active_after_perturbation": not torch.equal(
                changed_node, reference_node
            ),
        }

    reference = GPTransBlock(256, 32, 8, 0.0, 0.0, 1.0).to(device).eval()
    candidate = ConditionalPairRecurrenceBlock(
        256, 32, 8, 0.0, 0.0, 1.0
    ).to(device).eval()
    _load_preserved_state(candidate, reference, "conditional_recurrence")
    reference_node, reference_pair = reference(node, pair, mask)
    candidate_node, candidate_pair = candidate(node, pair, mask)
    if not torch.equal(reference_node, candidate_node) or not torch.equal(
        reference_pair, candidate_pair
    ):
        raise RuntimeError("Recurrence gate is not reference-equivalent at initialization")
    with torch.no_grad():
        candidate.conditional_recurrence.weight.fill_(0.01)
    _, changed_pair = candidate(node, pair, mask)
    return {
        "initial_output_bitwise_equal": True,
        "preserved_pair_to_node_readback": True,
        "conditional_recurrence_active_after_perturbation": not torch.equal(
            changed_pair, reference_pair
        ),
    }
