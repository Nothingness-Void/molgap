"""One parameter-free update-norm intervention on the retained BondLocal core."""
from .gptrans_bond_local import make_bond_local_model
from .gptrans_variants import normalized_pair_block


def construct_family_model(recipe, state):
    model = make_bond_local_model()
    for block in model.blocks:
        block.core = normalized_pair_block(block.core, "pair_update_norm")
    return model


def policy(mode):
    from .family_training_policy import FamilyModePolicy
    if mode != "degree_bond_local_pair_update_norm_ema999":
        raise ValueError("Unknown BondLocal pair normalization mode")
    return FamilyModePolicy(construct=construct_family_model,
                            objective_name="degree_bond_local_ema999")
