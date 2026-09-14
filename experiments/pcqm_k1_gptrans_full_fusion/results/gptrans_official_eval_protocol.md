# Independent GPTrans Scoring, 2026-09-14

The user requested GPTrans accuracy evaluation without waiting for K1. CPU job
1507573.ccpbs1 was submitted independently and observed running on ccc235.
It uses the already accepted final EMA model bundle and exactly the 73,545
official-valid rows in the existing accepted graph cache. It fits no parameters,
selects no checkpoint and does not reopen training or architecture search.

Execution uses CPU FP32, deterministic algorithms, batch 128 and eight Torch
threads within a 16-core allocation. Predictions are written atomically per
accepted graph shard; resume validates model/code/graph identity and labels.
A separate no-inference invocation reopens the persisted files, checks their
row identities, labels and hashes, and recomputes all MAEs before acceptance.

The reference is accepted EdgeState convergence official-valid Gap MAE
0.09963829815387726 eV on the same full validation role. This is a delivered
model comparison, not an equal-budget architecture ablation. CPU inference is
recorded explicitly. No official test-dev/challenge data is accessed.

Implementation commit: 13dd61a. Two focused tests passed, including synthetic
CPU forward execution and rejection of changed prediction labels/identity.
Training initialization hashes are not required to instantiate an inference
model: the complete hash-verified trained state is loaded strictly; frozen
architecture source and parameter count are still verified.

The existing GPU fusion study remains separately submitted. Its calibration
rule is unchanged; these standalone results do not authorize model tuning.
Exact job and source hashes: [submission](gptrans_official_eval_submission.json).
