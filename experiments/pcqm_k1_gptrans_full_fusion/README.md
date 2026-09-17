# Full-train K1 and GPTrans-T comparison

This experiment trains the frozen Neural-Atom K1 and GPTrans-T candidates on
the same accepted 3,378,606-row official training role. Each receives exactly
20,000,000 sample presentations, FP32, seed 42, batch 128, and no partial
batches. Separate outputs are retained; base training reads no validation/test
role.

- K1 contract: `../pcqm_k1_full/training_contract.json`
- GPTrans-T contract: `training_contract.json`
- Execution rationale and fusion protocol: `protocol.md`
- CLI entrypoints: `../pcqm_k1_full/run.py` and `run_gptrans_full.py`
- Fusion entrypoint: `run_fusion_study.py`
- PBS templates: `jobs/`
- Recovery and accepted training evidence (2026-09-14):
  `results/recovery_20260914_r3.md` and `results/gptrans_full_acceptance_r3.json`
- Independent official-valid evaluation: `results/gptrans_official_eval_protocol.md`
  and `results/gptrans_official_eval_submission.json`
- Integration decision: `decision.md`
- Accepted K1/fusion evidence and scientific decision (2026-09-15):
  `results/accepted_k1_gptrans_fusion_r3/decision.md`

Full training starts only after each model's A100 preflight passes. Every
checkpoint is atomic and contains the exact resumable cursor and RNG state.
After both base-model acceptance jobs pass, one fusion study runs on the
accepted official-valid graph cache. It fits one convex blend coefficient on
a frozen 20% calibration partition and evaluates on the remaining 80%, along
with fixed 50:50 and single-model comparisons. This consumes the official
validation role for one experiment; it does not use test-dev/challenge-test and
does not constitute a leaderboard score.
