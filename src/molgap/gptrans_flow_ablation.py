"""Parameter-preserving ablations of GPTrans node/pair propagation paths."""
from __future__ import annotations

import torch

from .gptrans import GPTransBlock, GraphPropagationAttention


MODES = ("no_pair_to_node", "no_pair_recurrence")


class NoPairToNodeAttention(GraphPropagationAttention):
    """Keep pair-biased attention and pair updates, but remove direct readback."""

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
        return self.output_dropout(self.output(node_update)), pair_update


class NoPairRecurrenceBlock(GPTransBlock):
    """Use each layer's pair message transiently without accumulating it."""

    def forward(self, node, pair, key_padding_mask):
        node_update, _pair_update = self.attention(
            self.node_norm1(node), pair, key_padding_mask
        )
        node = node + self.drop_path(self.attention_scale * node_update)
        node = node + self.drop_path(self.ffn_scale * self.ffn(self.node_norm2(node)))
        return node, pair


def _replace_attention(block):
    original = block.attention
    replacement = NoPairToNodeAttention(
        original.qkv.in_features,
        original.pair_to_attention.in_channels,
        original.num_heads,
        original.attention_dropout.p,
    )
    replacement.load_state_dict(original.state_dict(), strict=True)
    block.attention = replacement


def _replace_block(model, index):
    original = model.blocks[index]
    channels = original.node_norm1.normalized_shape[0]
    replacement = NoPairRecurrenceBlock(
        channels,
        original.attention.pair_to_attention.in_channels,
        original.attention.num_heads,
        original.attention.attention_dropout.p,
        original.drop_path.probability,
        1.0,
    )
    replacement.load_state_dict(original.state_dict(), strict=True)
    model.blocks[index] = replacement


def apply_flow_ablation(model, variant: str):
    if variant not in MODES:
        raise ValueError(variant)
    # Replacement construction must not advance the frozen training RNG stream.
    with torch.random.fork_rng(devices=[]):
        if variant == "no_pair_to_node":
            for block in model.blocks:
                _replace_attention(block)
        else:
            for index in range(len(model.blocks)):
                _replace_block(model, index)
    return model


def check_flow_mechanism(variant: str, device: str = "cuda") -> dict:
    if variant not in MODES:
        raise ValueError(variant)
    torch.manual_seed(42)
    mask = torch.tensor([[[[False, False, False, True]]]], device=device)
    node = torch.randn(1, 4, 256, device=device)
    pair = torch.randn(1, 32, 4, 4, device=device)

    if variant == "no_pair_to_node":
        reference = GraphPropagationAttention(256, 32, 8, 0.0).to(device).eval()
        candidate = NoPairToNodeAttention(256, 32, 8, 0.0).to(device).eval()
        candidate.load_state_dict(reference.state_dict(), strict=True)
        ref_node, ref_pair = reference(node, pair, mask)
        candidate_node, candidate_pair = candidate(node, pair, mask)
        assert torch.equal(ref_pair, candidate_pair)
        assert not torch.equal(ref_node, candidate_node)
        return {"pair_update_preserved": True, "direct_pair_to_node_removed": True}

    reference = GPTransBlock(256, 32, 8, 0.0, 0.0, 1.0).to(device).eval()
    candidate = NoPairRecurrenceBlock(256, 32, 8, 0.0, 0.0, 1.0).to(device).eval()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    ref_node, ref_pair = reference(node, pair, mask)
    candidate_node, candidate_pair = candidate(node, pair, mask)
    assert torch.equal(ref_node, candidate_node)
    assert torch.equal(candidate_pair, pair)
    assert not torch.equal(ref_pair, candidate_pair)
    return {"transient_node_update_preserved": True, "pair_recurrence_removed": True}
