"""Isolated, parameter-free relation-flow hypotheses on the frozen GPTrans core."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .gptrans import GraphPropagationAttention


MODES = ("pair_prenorm", "centered_logits")


def normalize_pair(pair):
    # Normalize channels independently for each pair, never across padding/nodes.
    return F.layer_norm(pair.permute(0, 2, 3, 1), (pair.shape[1],)).permute(0, 3, 1, 2)


def center_valid_logits(logits, key_padding_mask):
    valid = (~key_padding_mask).to(logits.dtype)
    offset = (logits * valid).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True).clamp_min(1)
    return logits - offset


class RelationFlowAttention(GraphPropagationAttention):
    def __init__(self, *args, variant: str, **kwargs):
        super().__init__(*args, **kwargs)
        if variant not in MODES:
            raise ValueError(variant)
        self.variant = variant

    def forward(self, node, pair, key_padding_mask):
        if self.variant == "pair_prenorm":
            return super().forward(node, normalize_pair(pair), key_padding_mask)

        batch_size, node_count, channels = node.shape
        qkv = self.qkv(node).reshape(
            batch_size, node_count, 3, self.num_heads, self.head_channels
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        logits = (query @ key.transpose(-2, -1)) * self.scale
        logits = logits + self.pair_to_attention(pair)
        # Node attention is unchanged; only the node-to-pair message is centered.
        residual_logits = center_valid_logits(logits, key_padding_mask)
        attention = self.attention_dropout(
            torch.softmax(logits.masked_fill(key_padding_mask, float("-inf")), dim=-1)
        )
        node_update = (attention @ value).transpose(1, 2).reshape(batch_size, node_count, channels)
        pair_update = self.attention_to_pair(attention + residual_logits)
        pair_weights = torch.softmax(pair_update.masked_fill(key_padding_mask, float("-inf")), dim=-1)
        pair_message = (pair_weights * pair_update).sum(-1).transpose(1, 2)
        node_update = node_update + self.pair_to_node(pair_message)
        return self.output_dropout(self.output(node_update)), pair_update


def apply_variant(model, variant: str):
    if variant == "reference":
        return model
    if variant not in MODES:
        raise ValueError(variant)
    # Extra module construction must not perturb the reference's training RNG.
    with torch.random.fork_rng(devices=[]):
        for block in model.blocks:
            original = block.attention
            replacement = RelationFlowAttention(256, 32, 8, 0.1, variant=variant)
            replacement.load_state_dict(original.state_dict(), strict=True)
            block.attention = replacement
    return model
