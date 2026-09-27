# Local inductive-bias pair attempt 002 release decision

The Kaggle1 v1 paired attempt ended during remote preflight: `rwse16` passed,
but `rwse16_local_edge` regenerated B-only random tensors with a full-state
hash different from the frozen local identity. The paired runner did not train
either arm. The [attempt record](../../platforms/_records/kaggle/training/pcqm_gptrans_local_inductive_bias_100k_s42_v1/attempt_001/README.md)
owns the raw observation. No new scientific result or replay entry was obtained.

The user authorized resubmission on 2026-09-27. Release at most one new
concurrent T4x2 attempt for the same `rwse16` and `rwse16_local_edge` question,
on a new immutable source dataset and kernel identity. The only source repair
loads a SHA-pinned B-only initial-state payload after constructing the model;
the verified base initialization, both full initial-state hashes, architecture,
data roles, FP32 recipe, seed, exposure, and frozen B-versus-A scientific gate
remain unchanged. This attempt does not retrain the historical GPTrans-T base.

Before push, require new source commit, package and Spec identities; a new
prospective trajectory for each arm with same-run replay binding; exact B addon
payload bytes and full-state hash; source/data mount checks; and the existing
real-shard model preflight. On Kaggle, both T4 runtime preflights must pass the
identity, repeatability, memory, and six-hour-per-arm gates before either arm
trains. Preserve separate checkpoints and replay evidence. The protected
official-validation and test roles remain sealed.

This authorization covers attempt 002 only. It does not authorize a later
retry, another seed, 500K/full training, protected-role use, or model promotion.
