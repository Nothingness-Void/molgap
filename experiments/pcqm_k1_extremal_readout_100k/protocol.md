# Frozen protocol

Authorized 2026-10-29: one Kaggle3 (nvoid912) T4x2 Notebook, fixed 100K, two arms, 40 passes
each, at most 14400 Python-allocation seconds including setup and 180 seconds cleanup, at
most 8 allocated T4-device hours. The authorization to spend kaggle3 quota and read 50K
development labels was relayed by the coordinating agent, not received directly from the
user. The ceilings were set by the coordinator; the user gave no specific numbers. No automatic
retry, continuation, monitor, protected roles, 500K/full or adoption.

## Evidence review and new question

Closed K1 families (2026-10-29 prior-art scan over experiments, server and archive refs):
initial residual (-0.708 meV), SAM (-2.834 meV), head-side inputs, normalization
(BN/LN/RMS/BRN/prenorm/exposure), dropout, DropPath (+2.373 meV), MsgNorm, hop-shell,
LayerScale/adaptive gain, degree balance, RWSE refresh, SSMA, K1-G, K4, slot96, atom256, FLAG,
consistency, EMA, tail weight decay, PNA-style neighbour statistics, longer-range exchange.
Readout neighbours: learned-query pooling on GPS9 (+6.936 meV, closed), K1 RepSet replacement
of mean pooling (+0.734 meV, closed), EdgeState multi-depth JK on QM9 (closed), mean+max on QM9
GPS9-192 (closed, different backbone). No K1 run keeps mean pooling and adds max, min, std or
second-order terms beside it (NOT_FOUND in all scanned trees; keyword scan, not an RML query).

Question: does a zero-initialised branch that adds per-graph max, min and std of the final
node states to the mean-pooled embedding improve the unchanged post-BN K1 at fixed 100K?

Papers (read through automated page summaries, not verified against raw text): Wang and Ji,
SOPool, arXiv:2007.10467 (second-order pooling helps GIN on small classification sets, no
regression, no max baseline); Buterez et al., arXiv:2211.04952 (learned readouts beat
sum/mean/max in 36 of 45 regression configurations on small molecular sets; max only a
baseline); Baek et al., GMT, arXiv:2102.11533 (attention pooling, classification);
Li et al., DeeperGCN, arXiv:2006.07739 (softmax/power-mean neighbour aggregators, node
tasks). None shows max/min or second-order readouts helping regression at 100K molecules.

Zero-GPU falsifiers (development role only, retained clean-BN EMA predictions with MAE 0.1431
eV, a different run from the paired baseline 0.1414): 5-fold cross-fit gradient boosting on the
signed residual, baseline 3.010 meV. Increment over baseline: raw attribute max/min/std +0.271,
raw covariance +0.459, both +0.531 meV; final-layer node-state max/min PCs +0.923, std PCs
+0.618, second-order +0.418, all final-layer statistics +1.253 meV (95% CI [1.09, 1.43]);
shuffled-row controls were negative. Probes are fitted on development labels, so they are
optimistic for post-hoc correction and do not bound what training with the branch learns.

Pre-registered bet: the frozen ceiling is about +1.25 meV against a 3 meV gate, so the prior of
clearing the gate is low (judgement, about 3-5%). A negative result closes the K1 extremal and
second-order readout family. The run is accepted mainly to time the pure-addon submit and
acceptance path.

## Matched arms and endpoint

`baseline` is preset k1-post-bn-100k-v1 with no addon (same-run reference). `candidate`
differs only by catalog addon `k1_extremal_readout`: a forward hook on mixer 9 stashes the final
node states, and a PRE_READOUT tap adds W [max, min, sqrt(var+1e-6)] to the pooled embedding,
with W a bias-free Linear(576 -> 192) initialised to zero. Statistics are per graph with the
shared graph count; max/min are masked scatter reductions with no infinite values. The head and
all backbone modules keep their state_dict keys, so the frozen k1.pt loads strictly. Seed 42,
BS128, FP32, LR 4e-4, WD 1e-5, 40-pass cosine to 1e-6, 31240 updates, best-development-live.

Primary gate: gain = MAE_baseline - MAE_candidate on the 50K development role, paired
2000-resample row bootstrap (seed 42). POSITIVE_UNDER_CONTRACT only if gain >= 3.0 meV and the
95% CI lower bound is above zero. Otherwise NEGATIVE_UNDER_CONTRACT, including a favourable
sub-gate gain. No retry, variant, seed or scale-up from this run. Row uncertainty is not
training-seed variance; one seed.

Descriptive readouts (not gates): norm of the projection W per statistic block at the selected
epoch (if the whole norm stays tiny the branch was unused); train-minus-dev gap per arm from
canonical traces; paired curve over 40 passes; gain by atom-count stratum via
experiment_attribution.analyze_family_pair.

## Roles, custody and implementation

Only the accepted official-train-derived 100K train and next 50K development roles.
Development has been reused for selection and is not an independent test. Official validation,
test-dev, test-challenge and common/OOD/P8-hard roles are not read.

Implementation: `src/molgap/k1_extremal_readout.py` and one catalog entry in
`experiment_catalog.py`. No change to trainer, optimizer step, loss, data pipeline or
execution modes.
