# Pair-memory dual 100K terminal decision

Decision date: 2026-09-24 (Asia/Tokyo).

Kaggle1 kernel `nothingnessvoid/molgap-gptrans-pair-memory-dual-100k-s42-v1` completed. The observed source commit, package identity, ExperimentSpec identity, graph manifest, fixed 100K/50K split, and source archive match the frozen release gate. Both arms passed no-inference mechanical acceptance, each with an accepted T4 runtime certificate, 60 epochs, 46,860 optimizer steps, 5,998,080 presentations, selected and resumable checkpoints, finite aligned predictions for all 50,000 development rows, and measured per-arm T4 cost. Official validation and test roles remain untouched.

`memory_value` scored 0.155583367 eV. `memory_message` scored 0.159906685 eV. The paired candidate-minus-reference delta is +0.004323316 eV (10,000-row bootstrap 95% interval +0.003265723 to +0.005336873 eV). `memory_message` therefore fails both the minimum 0.003 eV gain and the below-zero interval requirement. Close the mechanism comparison as `NEGATIVE_UNDER_CONTRACT`; no scale release follows.

The run is mechanically complete but not dual replay-ready. The retained per-epoch trace reports per-epoch steps and presentations rather than cumulative optimizer-step and sample-presentation coordinates. It also lacks observed live-development metrics and per-epoch checkpoint identities. Those values cannot be recovered by deriving them from epochs or batch size. Both traces are retained as 60-observation canonical traces on the epoch axis, with replay eligibility explicitly false. The strict V5 comparison-readiness gate is blocked by the absent trace fields; the replay pool must not claim complete entries for this run. No successor is authorized.

Evidence: `results/paired_comparison.json`, per-arm `results/*/mechanical_acceptance.json`, per-arm `prospective_replay/*/rml_finalized/`, and the raw remote artifacts under `platforms/_records/kaggle/training/pcqm_gptrans_pair_memory_dual_100k_kaggle1_v1/raw/`.
