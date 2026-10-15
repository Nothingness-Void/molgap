"""Reviewed family/addon declarations only; parser and IO live in common."""
from types import MappingProxyType
from .common.spec import (
    FamilyContract, AddonContract, AddonConfigField,
    EDGE_STATE_BASE_LAYERS, EDGE_STATE_MIN_LAYERS, EDGE_STATE_MAX_LAYERS,
)


FAMILIES = MappingProxyType({
    ("gptrans_scale_ema", "1"): FamilyContract(
        "gptrans_scale_ema", "1", "molgap.gptrans_scale_ema", "gptrans_scale_ema_v1",
        "ogb-atom9-bond3-shortest-path-cap20", ("train", "development"),
        "seed42-continuous-fixed500k-step-v1", "fixed-train-100k-mean-sample-std"),
    ("gptrans_scale_profile", "1"): FamilyContract(
        "gptrans_scale_profile", "1", "molgap.gptrans_scale_profile",
        "gptrans_scale_profile_v1", "ogb-atom9-bond3-shortest-path-cap20",
        ("train",), "seed42-step-fixed500k-profiling-v1",
        "fixed-train-100k-mean-sample-std",
    ),
    ("neural_atom_k1", "3"): FamilyContract(
        "neural_atom_k1", "3", "molgap.family_training_runner", "shared-family-training",
        "ogb-atom9-bond3-rwse16-v1", ("train",), "frozen-family-pass-order",
        "frozen-train-mean-sample-std", "ogb_edge_state_structural_gps9", ("development",), allow_any_seed=True),
    ("gptrans_t", "2"): FamilyContract(
        "gptrans_t", "2", "molgap.family_training_runner", "shared-family-training",
        "ogb-atom9-bond3-shortest-path-cap20", ("train",), "frozen-family-pass-order",
        "frozen-train-mean-sample-std", optional_roles=("development",), allow_any_seed=True),
    ("edge_state_gps", "1"): FamilyContract(
        "edge_state_gps", "1", "molgap.edge_state_model_only_v1",
        "edge_state_model_only_v1", "ogb-atom9-bond3-rwse16-v1",
        ("train", "development"), "seed42-epoch-global-randperm-v1",
        "train-mean-unbiased-std", "ogb_edge_state_structural_gps9",
    ),
    ("gptrans_t", "1"): FamilyContract(
        "gptrans_t", "1", "molgap.gptrans", "pcqm_gptrans_v4",
        "ogb-atom9-bond3-shortest-path-cap20", ("train", "development"),
        "seed-plus-epoch-global-randperm-v1", "fixed-train-100k-mean-sample-std",
    ),
    ("neural_atom_k1", "1"): FamilyContract(
        "neural_atom_k1", "1", "molgap.qm9_neural_atom", "pcqm_k1_full",
        "ogb-atom9-bond3-rwse16-v1", ("train",),
        "seed42-global-randperm-by-pass-v1", "full-train-mean-sample-std",
        "ogb_edge_state_structural_gps9",
    ),
    ("neural_atom_k1", "2"): FamilyContract(
        "neural_atom_k1", "2", "molgap.k1_screen_training", "pcqm_k1_v4_100k",
        "ogb-atom9-bond3-rwse16-v1", ("train", "development"),
        "seed42-python-epoch-shuffle-v4", "fixed-train-100k-mean-sample-std",
        "ogb_edge_state_structural_gps9",
    ),
})


# Each family's replacement group is exclusive; stacking is not supported.
ADDONS = MappingProxyType({
    **{(name, "1"): AddonContract("gptrans_t", "aggregate-research", "molgap.gptrans_research_campaign",
        family_versions=("2",), single_addon=True) for name in (
        "degree_aggregate_shared_ema999", "degree_aggregate_cycle_ema999",
        "degree_aggregate_carry_ema999", "degree_aggregate_correspondence_ema999")},
    ("degree_local_cycle_common_ema999", "1"): AddonContract(
        "gptrans_t", "cycle-pair-input", "molgap.gptrans_cycle_pair", family_versions=("2",), single_addon=True),
    **{(name, "1"): AddonContract("gptrans_t", "triplet-routing", "molgap.gptrans_triplet_routing", single_addon=True)
       for name in ("degree_local_triplet_product_v2_ema999", "degree_local_triplet_bond_route_ema999")},
    **{(name, "1"): AddonContract("gptrans_t", "triplet-followup", "molgap.gptrans_triplet_followup", single_addon=True)
       for name in ("degree_local_triplet_drop_ema999", "degree_local_triplet_product_ema999")},
    **{(name, "1"): AddonContract("gptrans_t", "triplet-communication", "molgap.gptrans_triplet_communication", single_addon=True)
       for name in ("degree_local_triplet_aggregate_ema999", "degree_local_triplet_attention_ema999")},
    **{(name, "1"): AddonContract("gptrans_t", "local-relation-flow", "molgap.gptrans_local_relation", single_addon=True)
       for name in ("degree_local_connected_pair_ema999", "degree_local_bond_return_ema999")},
    ("degree_bond_local_cap_ema999", "1"): AddonContract(
        "gptrans_t", "local-update-control", "molgap.gptrans_local_control", single_addon=True),
    ("degree_pair_transition_ema999", "1"): AddonContract(
        "gptrans_t", "relation-transition", "molgap.gptrans_pair_transition", single_addon=True),
    **{(name, "1"): AddonContract("gptrans_t", "capacity-intervention", "molgap.gptrans_capacity")
       for name in ("degree_node352_ema999", "degree_pair64_ema999", "degree_ffn2_ema999", "degree_bond_local_ema999")},
    ("edge_state_depth", "1"): AddonContract(
        "edge_state_gps", "edge-state-architecture", "molgap.edge_state_model_only_v1",
        (AddonConfigField("num_layers", "integer", minimum=EDGE_STATE_MIN_LAYERS,
                          maximum=EDGE_STATE_MAX_LAYERS, excluded=(EDGE_STATE_BASE_LAYERS,)),),
    ),
    ("neural_atom_k1", "1"): AddonContract(
        "edge_state_gps", "edge-state-architecture", "molgap.qm9_neural_atom",
    ),
    **{
        (name, "1"): AddonContract("gptrans_t", "attention-replacement", "molgap.gptrans_variants")
        for name in ("pair_prenorm", "centered_logits")
    },
    **{
        (name, "1"): AddonContract("gptrans_t", "attention-replacement", "molgap.gptrans_memory")
        for name in ("memory_value", "memory_message")
    },
    **{
        (name, "1"): AddonContract("gptrans_t", "input-intervention", "molgap.gptrans_author_variants")
        for name in ("degree_scale", "path_bond_mean", "degree_path_bond_mean", "degree_scale_ema999",
                     "degree_group_decay_ema999", "degree_path_endpoints_ema999", "degree_pair_depth_scale_ema999", "degree_path_bond_mean_ema999", "degree_decay001_ema999")
    },
    ("k1_pair_value", "1"): AddonContract(
        "neural_atom_k1", "pair-token-replacement", "molgap.k1_pair_token",
    ),
    **{(name, "1"): AddonContract("gptrans_t", "readout-intervention", "molgap.gptrans_readout")
       for name in ("degree_node_mean_readout_ema999", "degree_bond_mean_readout_ema999")},
    ("k1_joint_aggregation", "1"): AddonContract(
        "neural_atom_k1", "k1-local-aggregation", "molgap.k1_joint_aggregation",
        tuple(AddonConfigField(name, value=value) for name, value in
              (("layer", 6), ("latent_channels", 64), ("kappa", 4),
               ("seed", 42), ("degree_policy", "original-sum-above-four"))), ("2", "3"), True,
    ),
    **{(name, "1"): AddonContract("neural_atom_k1", "k1-training-view",
        "molgap.k1_screen_training", family_versions=("3",), single_addon=True)
       for name in ("k1_two_pass_mean", "k1_mean2_clean_second")},
})
