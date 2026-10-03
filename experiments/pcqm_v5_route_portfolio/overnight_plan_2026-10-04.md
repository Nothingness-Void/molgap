# G1 EMA portability — proposed bounded overnight diagnostic

Planning decision date: 2026-10-04, Asia/Tokyo. Owner: server, Track C.
Status: proposed; no compute release, prospective execution record, remote
attempt, source package, or heartbeat activation was created by this review.

## Why this question survived

The [local metadata review](results/review_2026-10-04.json) found no unresolved
terminal event or healthy active run in the five registered server control
states. The exact readout/combination binding was closed; desktop jobs were
outside the review. RML validation and frozen-output checks passed.

The accepted [EMA intervention](../pcqm_gptrans_input_ema_100k/gpu/results/decision.md)
improved G1 by 0.0066522625 eV on its original 50K development rows. All 60
live-optimization observations matched: this was a weight-averaging/selection
benefit, not a new encoder or more exposure. In contrast, chemical-path mean,
endpoint-path injection, grouped weight decay, uniform pair-depth scaling,
path/EMA combination, and final mean-readout substitutions did not establish
an incremental winner against the qualified comparator. See their owning
[input](../pcqm_gptrans_input_ema_100k/gpu/results/decision.md),
[recipe](../pcqm_gptrans_recipe_paths_100k/gpu/results/decision.md),
[pair-scale](../pcqm_gptrans_pair_scale_100k/gpu/results/decision.md),
[combination](../pcqm_gptrans_path_ema_combination_100k/gpu/results/decision.md), and
[readout](../pcqm_gptrans_readout_100k/gpu/results/decision.md) decisions.

The mean readouts fitted training data more closely but worsened development
endpoints. This was evidence against those replacements, not proof that the
relation backbone lacks capacity or needs an arbitrarily stronger regularizer.
The [author propagation code](https://github.com/czczup/GPTrans/blob/main/models/gptrans.py)
also accumulates pair updates outside the node DropPath operation. Its presence
in the local model therefore was not diagnosed as an accidental implementation
omission. No new pair-dropout experiment was released on that assumption.

The [earlier PairToken frozen audit](../pcqm_k1_cross_scale_frozen/decision.md)
showed that a selected 100K advantage could mostly disappear on different
molecules before any 500K training. That makes portability of the materially
positive EMA result a decision-relevant missing contrast. Repeating closed
readout/path modules or extending their budgets would not fill this gap.

## Hypothesis card

| Field | Declaration |
|---|---|
| Hypothesis | Shorter EMA averaging retained a benefit on the accepted, disjoint fixed500K development cohort, not solely on the development rows used for checkpoint selection. |
| Comparator | Accepted G1 selected EMA-0.9999 checkpoint. |
| Candidate | Accepted identical-architecture G1 selected EMA-0.999 checkpoint. |
| Changed quantity | Retained checkpoint/EMA averaging and its already frozen selection; no new architecture, loss, optimizer, seed, or training. |
| Alternative explanation | The original benefit was tied to selection cohort or finite-horizon averaging and may attenuate under different molecules or a larger training horizon. |
| Cheapest falsifier | Original-role checkpoint reproduction followed by paired frozen inference on the accepted fixed500K development role. |
| Decision affected | Whether to propose a separately authorized, matched 500K EMA training comparison; not full-scale promotion. |

## Frozen inputs and roles

The exact two checkpoint and retained prediction locators, actual file hashes,
reference bundles, target-transform pointer, and architecture identity were
verified without model loading in [the input inventory](results/review_2026-10-04.json).
Both models have 5,246,817 parameters and the same G1 architecture.

Use only these existing accepted roles:

| Role | Source indices, half-open | Purpose |
|---|---|---|
| fixed100K development | [100000, 150000) | Reproduce each saved original prediction payload. |
| fixed500K development | [500000, 550000) | Frozen-weight portability diagnostic. |

Dataset identities and accepted development-shard hashes are owned by the
[fixed-dataset acceptance](../../platforms/_records/kaggle/pcqm_fixed_datasets_v1/acceptance.json)
and the existing [role loader constants](../../src/molgap/pcqm_k1_cross_scale_diagnostic.py).
Do not create another split, random 10K sample, platform-specific cache, or
target normalization. Neither role is claimed to be an untouched final test;
the fixed500K development role has already been used for research selection.

Consume the original 100K training-only portable target transform unchanged.
No training shard, official validation, test-dev, test-challenge, external
dataset, teacher, ETKDG computation, or new geometry cache may be opened.
Geometry fields in accepted cached graphs remain unused by the pure-2D encoder.

## Proposed execution

Prefer Kaggle2's already accepted 100K/500K mirrors, subject to actual account
access and quota reconciliation at release. Their exact authority is the
[Kaggle2 mirror acceptance](../../platforms/_records/kaggle/pcqm_fixed_datasets_kaggle2_v1/acceptance.json).
Kaggle3 is an alternative only after access to the same byte-identical 500K
asset is qualified. The retained Kaggle3 metadata inspected in this review
described a 100K publication, not proof of a qualified 500K mount. Do not use
Kaggle1 or SCNet/IMS merely to bypass a missing dataset/account qualification.

Request `NvidiaTeslaT4`; isolate one device per worker before CUDA import.
Worker A loads the frozen old-EMA checkpoint; worker B loads the corrected-EMA
checkpoint. Each has its own model, output directory, runtime record and RNG.
FP32, TF32 disabled, deterministic execution and physical inference BS128 are
required. Include the final shorter inference batch; do not discard role rows.
Construct no optimizer, scheduler, EMA updater or training loader.

1. CPU preparation verifies the two checkpoint bytes, transform, actual
   accepted graph manifests and named development-shard bytes. It packages
   source/weights and exact runtime/mount identities using the existing owners.
2. Both workers reproduce their entire original 50K prediction payloads:
   exact source indices and targets, finite outputs, maximum absolute prediction
   discrepancy <=0.001 eV and whole-role MAE discrepancy <=0.0001 eV.
3. A shared gate opens fixed500K only after **both** original-role checks pass.
   A checkpoint, role, precision, target-transform or reproduction failure stops
   the physical diagnostic; no tolerance relaxation or inference on the new
   cohort follows.
4. Infer both frozen models on all 50K fixed500K development rows. Emit atomic,
   resumable 5K-row chunks, progress, role events and SHA-bound terminal manifests.
   Resume only verified unchanged-checkpoint chunks; never average or refit models.
5. Download only required artifacts, independently accept alignment/hash/runtime/
   role/cost/reproduction checks, then calculate paired errors and intervals.

Estimated GPU execution/setup: 30–60 minutes, **not yet a measured audit runtime**.
Proposed hard cap: 90 wall-minutes / 3 allocated T4-device hours for the dual,
including startup, preflight and any reserved idle device. CPU preparation and
queue/wall costs are recorded separately, not invented or converted into GPU
hours. A remote preflight must qualify feasibility before a full audit; a cap
termination produces partial evidence, never a fabricated complete result.

## Evidence and interpretation

Report each role's two MAEs, old-minus-corrected paired gain, molecule win
fraction, paired bootstrap with 5,000 resamples/seed 20260912, and the difference
in gains between cohorts. Define retained fraction as new-cohort gain divided
by the already observed original-role gain. An across-cohort contrast is not a
unique causal decomposition of distribution shift or checkpoint selection.

The proposed operational nomination rule is a positive fixed500K gain retaining
at least 50% of the original benefit (about 0.003326 eV), with the paired gain's
95% interval above zero. This is a **study-specific decision threshold**, not a
V5-wide scientific threshold, seed-variance estimate, or automatic promotion.
Freeze its machine-readable form before release; do not adjust it after seeing
the new predictions. Failure closes the audit without buying a 500K training
extension. A pass only justifies proposing the separate training contrast.

This is `NO_TRAIN` / diagnostic evidence. Its new-role prediction comparison
can be `PAIRED_ENDPOINT`; noncausal purpose cannot receive `STRICT_CAUSAL`.
The original training comparison retains its own strict/complete Replay status.
The audit does not prove that EMA0.999 will improve a genuinely 500K-trained
model: steps per pass and averaging windows change with scale.

Full RML closure still records prospective identity -> inference chunk/event
trace -> artifacts -> roles -> native costs -> acceptance -> terminal decision.
Training membership and external submission are not applicable; prediction
input, label read and metric computation are applicable on both named roles.
No checkpoint is selected using this audit, so its selection-use event is not
applicable; prior selection history remains attached. Do not invent training
steps or call an inference-only trajectory a new replay-ready training pair.

No automatic extra seed, 500K training, full training or protected-role action
is released. After submission, the existing Luna B binding/30-minute heartbeat
would be rebound to the actual returned identity. Healthy states remain silent;
only an idempotent terminal/fault handoff wakes the existing server A. Until
then the paused monitor stays paused.

## Reuse plan and outstanding release work

Task: frozen-checkpoint inference followed by saved-artifact acceptance.

Reuse accepted graph-role/hash validation and 5K chunk patterns from
`pcqm_k1_cross_scale_diagnostic.py` / `k1_portability_audit.py`; use GPTrans's
own model/forward/input semantics from `gptrans.py` / `pcqm_gptrans_v4.py`.
Reuse `training_reproducibility` IO/hashes, shared source/release/receipt
preflight, owning Kaggle adapter, saved-prediction paired analysis, RML terminal
APIs, and the existing server A/B control store. No new scheduler/framework.

The exact unsupported behavior is a GPTrans frozen-inference adapter with an
explicit role/range and chunk-resume gate. The training `_evaluate()` helper
hard-codes original-100K development indices and is not valid on fixed500K
as-is. The K1 audit model factory and target-transform SHA are likewise not
GPTrans defaults. Add only this narrow owning adapter before execution; do not
copy a trainer or falsely claim that registry declaration qualifies it.

Load the retained final state strictly into the matching GPTrans architecture.
Do not run `apply_author_variant()` after checkpoint loading: rescaling the
already learned degree tables again would change the frozen model. Do not
average the retained EMA weights again or recompute target statistics.

Before any POST: focused synthetic range/hash/finite/chunk/reproduction/barrier
tests, private input publication qualification, frozen source and prospective
noncausal record, `check-release` against actual staged mounts/entrypoint,
actual runtime qualification, and one-shot adapter/receipt binding must pass.
No future source commit, artifact, completed cost or result is asserted here.
