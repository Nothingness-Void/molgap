# Matched 500K V4 evidence bridge

Desktop authorization: 2026-09-15. Frozen source base: server 70500c5.

The three frozen architectures are OGB-rich EdgeState Structural GPS9
(`full_gps`, 4,771,073 parameters), original K1 (`neural_atom_k1`, 3,658,817),
and adapted GPTrans-T core (`gptrans`, 5,246,817). No geometry inputs or
pretraining are added. This is a new matched-budget benchmark, not reproduction
of each architecture's separately tuned historical score or a convergence claim.

All arms consume the accepted fixed 500K/50K official-train-derived assets.
The exact executable contract is `molgap.pcqm_500k_v4_evidence.scientific_contract`:
seed42, deterministic FP32/no TF32, physical BS128, 60 epochs, 3,906 steps per
epoch, 29,998,080 presentations, normalized L1, unfused/non-foreach AdamW,
weight decay 1e-5, clip1, shared cosine LR 4e-4 to 1e-6, raw-model development
selection. Each epoch drops 32 rows after a seeded global permutation.
Target normalization reads all 500K training labels only.

Each arm must pass optimizer-inclusive deterministic replay and memory reserve
before training. Runtime certificates and complete software identities are
retained. Cross-platform equivalence is assessed through the V4 validator;
the certificate is not a guarantee of identical final training trajectories.

Use Kaggle1 only. One T4x2 kernel isolates EdgeState and K1 into separate
processes/devices; another T4 allocation runs GPTrans on one isolated device.
The initial P100 request received T4x2 and exited before training; its successor
explicitly requests T4x2. Physical BS128 is invariant.
Each invocation requests the full remaining epoch budget and ends only when the
60-epoch contract completes or the projected next epoch approaches the Kaggle
session limit. The original four-epoch staging policy was an execution mistake
and was retired after epoch 16; it added queue and publication latency without
changing the scientific comparison. Each invocation publishes best/last,
RNG, optimizer, deterministic schedule cursor, initial state, all epoch
development predictions, timing, and manifests. The next stage must mount a
new private dataset containing the previous accepted outputs and verify its
hashes. No in-place source replacement or scratch restart on missing resume.

Training stage completion is not final acceptance. Only after all arms reach
the frozen exposure, compare paired residuals and development MAE with row
bootstrap intervals. A gain of 0.003 eV is the nomination floor, not an assumed
measured noise floor. Another seed or independent once-read role requires a
separately frozen decision. Official validation/test and shadow roles are sealed.

Historical paired-v3 500K assets remain contextual because of tail batches or
different optimizer/schedule/selection contracts. Do not relabel them as V4.
