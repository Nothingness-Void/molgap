# Promotion 2026-11-17: molgap-desktop to master

Master is stable delivery. This promotion copies reviewed, reusable work from
`molgap-desktop` (tip `e53120da`). It is a selected copy, not a merge: no
history of desktop is on master. Branch: `codex/promote/desktop-to-master-20261117`.

## Promoted

| Item | What | Source |
|---|---|---|
| `src/molgap/` | The whole library, including K1 and GPT-T research code and registry updates. Master keeps 29 older modules that desktop moved to the `archive` branch (`hybrid.py`, `late_router.py`, `multi2d_data.py`, `schnetpack.py`, `visnet.py`, `archive/`), because master's historical scripts still import them. | desktop `e53120da` |
| `pyproject.toml` | pytest settings (`testpaths`, `pythonpath`). | desktop `e53120da` |
| `tests/` | 136 test files that pass without the excluded folders. Master's 2 older tests that the new library made stale were removed. | desktop `e53120da` |
| `models/README.md`, `models/REFERENCE_INDEX.md` | Asset map and reference roles. | desktop `e53120da` |
| `CURRENT_STATE.md` | Track B section rewritten with closed, verified results and source hashes. | desktop `CURRENT_STATE.md` plus the decision files named in it |
| `ARCHITECTURE.md`, `README.md` | A short note on the research library; the Delta/UQ line now says the helpers need an `archive` branch checkout and a historical v3 base. | this branch |

## Not promoted

- `experiments/`, `platforms/` (including `platforms/_records`), `research_memory/` data, `docs/operations/`, `BRANCHES.md`, desktop `AGENTS.md`/`ROADMAP.md`: they are operations records and workflow, not stable delivery.
- Deletions that desktop made under `production/`. Master's `production/` is unchanged.
- Large files and `.pt` weights.
- 70 test files. 53 fail here because they read `experiments/`, `platforms/`, `docs/operations/` or `BRANCHES.md`. 17 more import helpers from those tests (or fail on an excluded sibling) and were removed too. All of them:

  `test_colab_v5_runtime`, `test_documentation_navigation`, `test_edge_state_training_core`, `test_execution_comparison`, `test_experiment_attribution`, `test_experiment_cli`, `test_experiment_db`, `test_experiment_family_workflow`, `test_experiment_launch`, `test_experiment_lifecycle_integration`, `test_experiment_prospective`, `test_experiment_run`, `test_experiment_terminal`, `test_experiment_terminal_staging`, `test_experiment_workflow`, `test_experiment_workflow_resume`, `test_family_commit_safety`, `test_family_recovery_guards`, `test_family_training_io`, `test_family_training_runner`, `test_fast_sandbox`, `test_gptrans_500k_frozen_readout`, `test_gptrans_b1024_runner`, `test_gptrans_execution_refine`, `test_gptrans_local_pair_analysis`, `test_gptrans_pair_norm_500k`, `test_gptrans_pair_norm_500k_profile`, `test_graph_screen_training`, `test_ims_v4_submit`, `test_k1_execution_profile`, `test_k1_frozen_recovery`, `test_k1_local_speed`, `test_k1_pair_preparation`, `test_kaggle_bootstrap`, `test_kaggle_family_retrieval`, `test_kaggle_quota_monitor`, `test_kaggle_submission`, `test_local_testing`, `test_modular_experiment_cli`, `test_pcqm_500k_v4_evidence`, `test_pcqm_distance_angle_scale`, `test_pcqm_fixed_datasets`, `test_pcqm_gptrans_v4`, `test_pcqm_official_edge_state`, `test_pcqm_scale_transfer_reassessment`, `test_pcqm_topology_preflight`, `test_reform_daily_platform`, `test_reform_distribution`, `test_reform_input`, `test_reform_retrieval`, `test_reform_review`, `test_reform_workflow`, `test_repository_layout`, `test_research_memory`, `test_run_acceptance_cli`, `test_run_acceptance_integration`, `test_run_acceptance_resolution`, `test_run_attribution`, `test_same_run_replay`, `test_scale_fit_retained`, `test_scientific_input`, `test_scientific_input_ergonomics`, `test_scientific_pair_input`, `test_shared_family_lifecycle`, `test_shared_family_registration`, `test_shared_graph_lifecycle`, `test_shared_resume_transport`, `test_shared_training_compatibility`, `test_terminal_trace_closure`, `test_v5_desktop_contract`

## Recover what was left out

- Desktop state: `git show molgap-desktop:<path>` or `git checkout e53120da -- <path>`.
- Excluded tests: `git checkout e53120da -- tests/<file>`, together with the folders they need (`experiments/`, `platforms/`).
- Experiment evidence for each Track B result is on the branch named in `CURRENT_STATE.md`:
  - `codex/exp/k1-muon-500k-20261106`, `6b824342`
  - `codex/exp/gptrans-full-levers-20261115`, `33ab4571`
  - `codex/exp/gptrans-full-minus-tail-20261110`, `e32009ae`
  - `codex/exp/gptrans-tokengt-bond-100k-20261117`, `1dc9c014`
  - `codex/exp/gptrans-width-500k-20261117`, `532be162`, `ac983785`
  - K1 Muon 100K decision: desktop `5039fa0c`.
- The TokenGT and width branches also change `src/molgap` (`gptrans_tokengt_bond.py`, `gptrans_variants.py`, `kaggle_accelerator_push.py`); those changes are not on master.

## Verification

`import molgap` and 136 test files (run one file at a time with `PYTHONPATH=src`) pass on the promote branch.
