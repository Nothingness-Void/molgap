"""Reviewed family/addon capabilities; parsing and IO live in common."""
from types import MappingProxyType
from .common.spec import (
    FamilyContract, FamilyExecution, AddonContract, AddonConfigField,
    EDGE_STATE_BASE_LAYERS, EDGE_STATE_MIN_LAYERS, EDGE_STATE_MAX_LAYERS,
)


FAMILIES = MappingProxyType({
    ("neural_atom_k1", "3"): FamilyContract(
        "neural_atom_k1", "3", "molgap.family_training_runner", "shared-family-training",
        "ogb-atom9-bond3-rwse16-v1", ("train",), "frozen-family-pass-order",
        "frozen-train-mean-sample-std", "ogb_edge_state_structural_gps9", ("development",), allow_any_seed=True,
        execution=FamilyExecution("molgap.family_training_runner", "k1-session-v1", ("molgap.pcqm_wedge",), target_identity_in_completion=True, profile_adapter="molgap.family_execution_profile"),
    ),
    ("gptrans_t", "2"): FamilyContract(
        "gptrans_t", "2", "molgap.family_training_runner", "shared-family-training",
        "ogb-atom9-bond3-shortest-path-cap20", ("train",), "frozen-family-pass-order",
        "frozen-train-mean-sample-std", optional_roles=("development",), allow_any_seed=True,
        execution=FamilyExecution("molgap.family_training_runner", "gptrans-session-v1", ("molgap.pcqm_wedge",), target_identity_in_completion=True, profile_adapter="molgap.family_execution_profile"),
    ),
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
        execution=FamilyExecution("molgap.gptrans_screen_workflow", "gptrans-v1", ("molgap.pcqm_wedge",)),
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
        replication_samplers=((43, "seed43-python-epoch-shuffle-v4"),),
        execution=FamilyExecution("molgap.k1_screen_training", "k1-screen-v1", ("molgap.pcqm_wedge",), target_identity_in_completion=True),
    ),
})


# Replacement groups are exclusive; independent registered slots may compose.
ADDONS = MappingProxyType({
    **{
        (prefix + "_" + suffix + "_scale", "1"): AddonContract(
            family, "slot-" + suffix, "molgap.substrate_scale_addon",
            tuple(AddonConfigField(key, value=value) for key, value in (
                ("family", family), ("slot", slot), ("name", suffix + "_scale"))),
            (version,),
            execution_modes=((version, "composed"),),
            apply_hook="molgap.substrate_scale_addon:apply_addon",
        )
        for family, prefix, version in (("neural_atom_k1", "k1", "3"), ("gptrans_t", "gptrans", "2"))
        for suffix, slot in (("node", "pre_message"), ("aggregate", "post_aggregation"),
                             ("readout", "pre_readout"))
    },
    ("gptrans_degree_bond_local", "1"): AddonContract(
        "gptrans_t", "local-bond-architecture", "molgap.gptrans_bond_local",
        family_versions=("2",), single_addon=True,
        execution_modes=(("2", "degree_bond_local_ema999"),), policy_hook="molgap.gptrans_bond_local:policy",
    ),
    ("edge_state_depth", "1"): AddonContract(
        "edge_state_gps", "edge-state-architecture", "molgap.edge_state_model_only_v1",
        (AddonConfigField("num_layers", "integer", minimum=EDGE_STATE_MIN_LAYERS,
                          maximum=EDGE_STATE_MAX_LAYERS, excluded=(EDGE_STATE_BASE_LAYERS,)),),
    ),
    ("neural_atom_k1", "1"): AddonContract(
        "edge_state_gps", "edge-state-architecture", "molgap.qm9_neural_atom",
    ),
    **{
        (name, "1"): AddonContract("gptrans_t", "attention-replacement", "molgap.gptrans_variants",
            execution_modes=(("1", name), ("2", name)),
        )
        for name in ("pair_prenorm", "centered_logits")
    },
    **{
        (name, "1"): AddonContract("gptrans_t", "attention-replacement", "molgap.gptrans_memory",
            execution_modes=(("1", name), ("2", name)),
        )
        for name in ("memory_value", "memory_message")
    },
    ("k1_pair_value", "1"): AddonContract(
        "neural_atom_k1", "pair-token-replacement", "molgap.k1_pair_token",
    ),
    ("k1_joint_aggregation", "1"): AddonContract(
        "neural_atom_k1", "k1-local-aggregation", "molgap.k1_joint_aggregation",
        (AddonConfigField("layer", value=6), AddonConfigField("latent_channels", value=64),
         AddonConfigField("kappa", value=4), AddonConfigField("seed", "initialization_seed", value=42),
         AddonConfigField("degree_policy", value="original-sum-above-four")),
        ("2", "3"), True,
        execution_modes=(("2", "ssma"), ("3", "ssma")),
    ),
    ("k1_two_pass_mean", "1"): AddonContract(
        "neural_atom_k1", "k1-training-objective", "molgap.k1_screen_training",
        family_versions=("2", "3"), single_addon=True,
        execution_modes=(("2", "mean2"), ("3", "mean2")),
    ),
    ("k1_mean2_clean_second", "1"): AddonContract(
        "neural_atom_k1", "k1-training-objective", "molgap.k1_screen_training",
        family_versions=("2", "3"), single_addon=True,
        execution_modes=(("2", "mean2_clean_second"), ("3", "mean2_clean_second")),
    ),
    ("k1_sam", "1"): AddonContract(
        "neural_atom_k1", "k1-training-objective", "molgap.family_model_ops",
        (AddonConfigField("rho", value=0.05),),
        ("3",), True,
        execution_modes=(("3", "sam"),), policy_hook="molgap.family_model_ops:policy",
    ),
    # Historical declaration compatibility only; live training dispatch retires these.
    ("k1_fused_layout", "1"): AddonContract(
        "neural_atom_k1", "k1-execution", "molgap.k1_screen_training",
        family_versions=("2",), single_addon=True,
    ),
    ("k1_initialized_layout", "1"): AddonContract(
        "neural_atom_k1", "k1-execution", "molgap.k1_screen_training",
        family_versions=("2",), single_addon=True,
    ),
})
