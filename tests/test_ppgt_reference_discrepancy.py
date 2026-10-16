"""A parameter inventory is not a fabricated upstream reproduction certificate."""
import torch

from molgap.ppgt_model import PPGT, PPGTConfig


def test_paper_small_dense_backbone_alone_exceeds_reported_whole_model():
    # The author source at 62cafd3 uses full-width Q, KV and O and a two-linear
    # FFN. This lower bound excludes every bias, norm, PE stem and graph head.
    d, expansion, blocks = 512, 2, 16
    lower_bound = blocks * (4 * d * d + 2 * expansion * d * d)
    assert lower_bound == 33_554_432 and lower_bound > 17_600_000
    # Meta construction counts the unchanged local model without allocating
    # the 35M reference tensors on this low-memory computer.
    with torch.device("meta"):
        model = PPGT(PPGTConfig(hidden=d, heads=16))
    inventory = {name: sum(p.numel() for p in child.parameters())
                 for name, child in model.named_children()}
    assert inventory["blocks"] == 33_710_592
    assert sum(inventory.values()) == 34_867_201
    assert len({id(block) for block in model.blocks}) == blocks
