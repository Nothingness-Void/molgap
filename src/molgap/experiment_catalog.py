"""Reviewed family/addon capabilities; host-neutral parsing lives in common.spec."""
from types import MappingProxyType
from .common.spec import (FamilyContract, FamilyExecution, AddonContract, AddonConfigField,
    EDGE_STATE_BASE_LAYERS, EDGE_STATE_MIN_LAYERS, EDGE_STATE_MAX_LAYERS)

FAMILIES = MappingProxyType({
    **{(name, version): FamilyContract(name, version, "molgap." + module,
        "modern2d_formal_training_v1", schema, ("train",),
        "frozen-family-pass-order", "frozen-train-mean-sample-std",
        optional_roles=("development",),
        execution=FamilyExecution("molgap.family_training_runner", "modern2d-formal-session-v1",
            model_factory="molgap." + module + ":make_model",
            target_identity_in_completion=True,
            profile_adapter="molgap.family_execution_profile",
            source_files=("src/molgap/modern2d_training.py", "src/molgap/modern2d_inputs.py",
                          "src/molgap/" + features + ".py", *extra_sources),
            scientific_input_abi="modern2d-indexed-v1",
            model_interface=(name, preview_version),
            feature_module="molgap." + features,
            feature_source_files=feature_source_files,
            feature_config_fields=feature_config_fields,
            input_model_name=module + "_v" + preview_version))
       for name, version, preview_version, module, features, schema, extra_sources, feature_source_files, feature_config_fields in (
           ("ppgt_small", "3", "2", "ppgt_author", "ppgt_author_features", "ogb-atom9-bond3-author-rrwp-spe-v2", (), (),
            ("rrwp_steps", "sym_norm", "spe_bases", "max_nodes", "max_edges")),
           ("esa_small", "4", "3", "esa_native", "esa_features", "ogb-atom9-bond3-directed-edge-isolate-mask-v3",
            ("src/molgap/esa_model.py", "src/molgap/esa_invariant.py"), ("src/molgap/esa_native.py",),
            ("max_nodes", "max_edges")))},
    **{(name, version): FamilyContract(name, version, "molgap." + module,
        "modern2d_session_preview_v1", schema, ("train",),
        "frozen-family-pass-order", "frozen-train-mean-sample-std",
        optional_roles=("development",),
        execution=FamilyExecution("molgap.family_training_runner", "modern2d-session-preview-v1",
            model_factory="molgap." + module + ":make_model",
            profile_adapter="molgap.family_execution_profile",
            source_files=("src/molgap/modern2d_inputs.py", "src/molgap/" + features + ".py",
                          *extra_sources),
            scientific_input_abi="modern2d-indexed-v1",
            model_interface=(name, version),
            feature_module="molgap." + features,
            feature_source_files=feature_source_files,
            feature_config_fields=feature_config_fields,
            input_model_name=module + "_v" + version))
       for name, version, module, features, schema, extra_sources, feature_source_files, feature_config_fields in (
           ("ppgt_small", "2", "ppgt_author", "ppgt_author_features", "ogb-atom9-bond3-author-rrwp-spe-v2", (), (),
            ("rrwp_steps", "sym_norm", "spe_bases", "max_nodes", "max_edges")),
           ("esa_small", "3", "esa_native", "esa_features", "ogb-atom9-bond3-directed-edge-isolate-mask-v3",
            ("src/molgap/esa_model.py", "src/molgap/esa_invariant.py"), ("src/molgap/esa_native.py",),
            ("max_nodes", "max_edges")))},
    ("esa_small", "2"): FamilyContract("esa_small", "2", "molgap.esa_invariant",
        "modern2d_cpu_v1", "ogb-atom9-bond3-directed-edge-mask-v1", ("train", "development"),
        "frozen-family-pass-order", "frozen-train-mean-sample-std",
        execution=FamilyExecution("molgap.family_training_runner","modern2d-cpu-v1",
            model_factory="molgap.esa_invariant:make_model",cpu_synthetic_only=True,
            source_files=("src/molgap/modern2d_inputs.py","src/molgap/esa_features.py","src/molgap/esa_model.py"))),
    **{(name, "1"): FamilyContract(name, "1", "molgap." + module,
        "modern2d_cpu_v1", schema, ("train", "development"),
        "frozen-family-pass-order", "frozen-train-mean-sample-std",
        execution=FamilyExecution("molgap.family_training_runner","modern2d-cpu-v1",
            model_factory="molgap."+module+":make_model",cpu_synthetic_only=True,
            source_files=("src/molgap/modern2d_inputs.py","src/molgap/"+features+".py")))
       for name, module, features, schema in (
           ("ppgt_small", "ppgt_model", "ppgt_features", "ogb-atom9-bond3-rrwp-spe-v1"),
           ("esa_small", "esa_model", "esa_features", "ogb-atom9-bond3-directed-edge-mask-v1"))},
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
        "frozen-train-mean-sample-std", "ogb_edge_state_structural_gps9", ("development",),
        execution=FamilyExecution("molgap.family_training_runner","k1-session-v1",("molgap.pcqm_wedge",),
            ("src/molgap/family_training_runner.py","src/molgap/family_training_kernel.py",
             "src/molgap/k1_screen_training.py","src/molgap/qm9_neural_atom.py",
             "src/molgap/k1_execution_layout.py","src/molgap/k1_clean_second.py"),
            target_identity_in_completion=True,profile_adapter="molgap.family_execution_profile",
            scientific_input_abi="shared-family-v1")),
    ("gptrans_t", "2"): FamilyContract(
        "gptrans_t", "2", "molgap.family_training_runner", "shared-family-training",
        "ogb-atom9-bond3-shortest-path-cap20", ("train",), "frozen-family-pass-order",
        "frozen-train-mean-sample-std", optional_roles=("development",),
        execution=FamilyExecution("molgap.family_training_runner","gptrans-session-v1",("molgap.pcqm_wedge",),
            ("src/molgap/family_training_runner.py","src/molgap/family_training_kernel.py",
             "src/molgap/pcqm_gptrans_v4.py","src/molgap/gptrans.py",
             "src/molgap/gptrans_capacity.py","src/molgap/gptrans_cycle_pair.py",
             "src/molgap/gptrans_cycle_inputs.py","src/molgap/pcqm_fixed_records.py",
             "src/molgap/gptrans_research_campaign.py","src/molgap/gptrans_triplet_communication.py",
             "src/molgap/gptrans_local_relation.py","src/molgap/bond_message_carry.py"),
            target_identity_in_completion=True,profile_adapter="molgap.family_execution_profile",
            scientific_input_abi="shared-family-v1")),
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
        execution=FamilyExecution("molgap.gptrans_screen_workflow","gptrans-v1",("molgap.pcqm_wedge",),
            ("src/molgap/gptrans_screen_workflow.py","src/molgap/pcqm_gptrans_v4.py",
             "src/molgap/gptrans.py","src/molgap/gptrans_variants.py","src/molgap/pcqm_wedge.py"),
            profile_adapter="molgap.gptrans_legacy_execution_profile"),
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
        execution=FamilyExecution("molgap.k1_screen_training","k1-screen-v1",("molgap.pcqm_wedge",),
            ("src/molgap/k1_screen_training.py","src/molgap/qm9_neural_atom.py",
             "src/molgap/qm9_local_hierarchy.py","src/molgap/qm9_gape.py",
             "src/molgap/pcqm_gap_architecture.py","src/molgap/gps.py","src/molgap/pcqm_wedge.py"),
            profile_adapter="molgap.k1_legacy_execution_profile"),
    ),
})


ADDONS = MappingProxyType({
    **{(name, "1"): AddonContract(family,
        "modern2d-structural-falsifier", "molgap.modern2d_structure_addons",
        (AddonConfigField("variant", value=variant),), (version,),
        execution_modes=((version, "composed"),),
        apply_hook="molgap.modern2d_structure_addons:apply_addon")
       for name, family, version, variant in (
           ("ppgt_content_pair", "ppgt_small", "3", "content_pair"),
           ("esa_count_readout", "esa_small", "4", "count_readout"))},
    **{("ppgt_research_" + variant, "1"): AddonContract("ppgt_small",
        "ppgt-research-mechanism", "molgap.ppgt_research_addons",
        (AddonConfigField("variant", value=variant),), ("3",),
        execution_modes=(("3", "composed"),),
        apply_hook="molgap.ppgt_research_addons:apply_addon")
       for variant in ("mean_readout", "residual_scale", "both")},
    ("esa_identity_residual", "1"): AddonContract("esa_small",
        "esa-research-residual", "molgap.esa_research_addons",
        (AddonConfigField("variant", value="identity_residual"),), ("4",),
        execution_modes=(("4", "composed"),),
        apply_hook="molgap.esa_research_addons:apply_addon"),
    **{(prefix + "_readout_scale", "1"): AddonContract(family,
        "slot-readout", "molgap.substrate_scale_addon",
        tuple(AddonConfigField(key, value=value) for key, value in (
            ("family", family), ("slot", "pre_readout"), ("name", "readout_scale"))),
        (version,), execution_modes=((version, "composed"),),
        apply_hook="molgap.substrate_scale_addon:apply_addon")
       for family, prefix, version in (("ppgt_small", "ppgt", "3"), ("esa_small", "esa", "4"))},
    **{(prefix + "_" + suffix + "_scale", "1"): AddonContract(family,
        "slot-" + suffix, "molgap.substrate_scale_addon",
        tuple(AddonConfigField(key, value=value) for key, value in (
            ("family", family), ("slot", slot), ("name", suffix + "_scale"))), (version,),
        execution_modes=((version,"composed"),), apply_hook="molgap.substrate_scale_addon:apply_addon")
       for family, prefix, version in (("neural_atom_k1", "k1", "3"), ("gptrans_t", "gptrans", "2"))
       for suffix, slot in (("node", "pre_message"), ("aggregate", "post_aggregation"), ("readout", "pre_readout"))},
    **{(name, "1"): AddonContract("gptrans_t", "aggregate-research", "molgap.gptrans_research_campaign",
        family_versions=("2",), single_addon=True, execution_modes=(("2",name),),
        policy_hook="molgap.gptrans_training_policy:policy") for name in (
        "degree_aggregate_shared_ema999", "degree_aggregate_cycle_ema999",
        "degree_aggregate_carry_ema999", "degree_aggregate_correspondence_ema999")},
    ("degree_local_cycle_common_ema999", "1"): AddonContract(
        "gptrans_t", "cycle-pair-input", "molgap.gptrans_cycle_pair", family_versions=("2",), single_addon=True,
        execution_modes=(("2","degree_local_cycle_common_ema999"),),policy_hook="molgap.gptrans_training_policy:policy"),
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
    **{(name, "1"): AddonContract("gptrans_t", "capacity-intervention", "molgap.gptrans_capacity",
        execution_modes=(("2",name),) if name=="degree_bond_local_ema999" else (),
        policy_hook="molgap.gptrans_training_policy:policy" if name=="degree_bond_local_ema999" else None)
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
        (name, "1"): AddonContract("gptrans_t", "attention-replacement", "molgap.gptrans_variants",
            execution_modes=(("1",name),("2",name)))
        for name in ("pair_prenorm", "centered_logits")
    },
    **{
        (name, "1"): AddonContract("gptrans_t", "attention-replacement", "molgap.gptrans_memory",
            execution_modes=(("1",name),("2",name)))
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
        execution_modes=(("2","ssma"),("3","ssma")),
         conditional_optimizer_prefixes=("k1_joint_aggregation.",),
         conditional_optimizer_buffers=("k1_joint_aggregation.affine_weight",
                                        "k1_joint_aggregation.affine_bias"),
         legacy_conditional_optimizer_tail=True,
    ),
    **{(name, "1"): AddonContract("neural_atom_k1", "k1-training-view",
        "molgap.k1_screen_training", family_versions=("3",), single_addon=True,
        execution_modes=(("3",mode),))
       for name,mode in (("k1_two_pass_mean","mean2"),("k1_mean2_clean_second","mean2_clean_second"))},
})
