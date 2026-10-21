# ESA LR saved-metadata audit

Status: METADATA_VERIFIED_ACCEPTANCE_PENDING
Checks: {"passed": 46, "failed": 0}
Errors: []
Missing: []

Train MAE is normalized-Gap, NOT eV; development MAE is eV. No cross-unit subtraction or completed-reference comparison.
Outcome dimensions: {"execution": "interrupted", "budget": "STOP_FOR_COST", "scientific": "INCONCLUSIVE", "comparison": "context_only"}

Arms: {"lr1e4": {"best_observed_dev_MAE_eV": 0.21137726306915283, "best_observed_pass": 27, "horizon_steps": 46860, "last_dev_MAE_eV": 0.21529927849769592, "last_observed_pass": 29, "last_observed_step": 22649, "last_train_normalized_Gap": 0.08143311036838917, "native_state_validation": "PENDING .pt readiness confirmation", "observations": 29, "paused_cursor": {"completed_passes": 29, "next_batch": 708}, "paused_optimizer_step": 23357, "profile_summary_sha256": "b08203de584312c1f2c0cf08573822ccd9db98e72b57d10c78616c012d92a6d8", "raw_trace_sha256_DECLARED_ONLY": "263a81818bbbd6f1a88524d49f7afa7dbed573b3e54fadccdbab875abddd0fac", "recipe_identity": "0acdc03a59ac59cf646d2ee370f0e30452b05ecd3962270b5bf53aa6f1de925e", "runtime_platform_version": null}, "lr3e4": {"best_observed_dev_MAE_eV": 0.20648691058158875, "best_observed_pass": 27, "horizon_steps": 46860, "last_dev_MAE_eV": 0.2099905014038086, "last_observed_pass": 29, "last_observed_step": 22649, "last_train_normalized_Gap": 0.06658266682211171, "native_state_validation": "PENDING .pt readiness confirmation", "observations": 29, "paused_cursor": {"completed_passes": 29, "next_batch": 685}, "paused_optimizer_step": 23334, "profile_summary_sha256": "7069c2dad59f3426b40cf3cb4265e1ee3ce6f4b729da1edcd0b8d5df39ec9bc1", "raw_trace_sha256_DECLARED_ONLY": "b264145c49b44086544b88a70eba0e08e69e1829064990f0ff5c34f18535a5a1", "recipe_identity": "9eaf295634f92458b3606ad9c3a231f83091e1d578f1819d6788842b46cfb580", "runtime_platform_version": null}}

Pending: Native/partial saved states awaiting main readiness confirmation; Raw Chrome trace bytes not independently checked in metadata audit; Interrupted run has no completed 60-pass output manifest; no full reference/scientific acceptance or Replay claim

JSON: E:\MolGap\artifacts\runs\esa-lr-attribution-100k-s42-v1\v1\inspection\independent\metadata-audit.json
