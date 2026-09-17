"""Direct persistent-pair readback, isolated from relation normalization."""
import torch

from .gptrans import GraphPropagationAttention

MODES = ("memory_value", "memory_message")


def readback(pair, delta, mask, variant):
    if variant not in MODES:
        raise ValueError(variant)
    memory = pair + delta
    routing = delta if variant == "memory_value" else memory
    weights = torch.softmax(routing.masked_fill(mask, float("-inf")), dim=-1)
    return (weights * memory).sum(-1).transpose(1, 2)


class MemoryAttention(GraphPropagationAttention):
    def __init__(self, *args, variant, **kwargs):
        super().__init__(*args, **kwargs)
        self.variant = variant

    def forward(self, node, pair, key_padding_mask):
        batch_size, node_count, channels = node.shape
        qkv = self.qkv(node).reshape(batch_size, node_count, 3, self.num_heads,
                                     self.head_channels).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        logits = (query @ key.transpose(-2, -1)) * self.scale
        logits = logits + self.pair_to_attention(pair)
        attention = self.attention_dropout(torch.softmax(
            logits.masked_fill(key_padding_mask, float("-inf")), dim=-1))
        node_update = (attention @ value).transpose(1, 2).reshape(batch_size, node_count, channels)
        pair_update = self.attention_to_pair(attention + logits)
        # Only this direct readback changes; GPTransBlock still adds pair_update once.
        message = readback(pair, pair_update, key_padding_mask, self.variant)
        node_update = node_update + self.pair_to_node(message)
        return self.output_dropout(self.output(node_update)), pair_update


def apply_memory_variant(model, variant):
    if variant not in MODES:
        raise ValueError(variant)
    with torch.random.fork_rng(devices=[]):
        for block in model.blocks:
            replacement = MemoryAttention(256, 32, 8, 0.1, variant=variant)
            replacement.load_state_dict(block.attention.state_dict(), strict=True)
            block.attention = replacement
    return model
