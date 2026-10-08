# Frozen K1 500K consistency ablation — 2026-10-07

Authority: user requested this ablation to obtain evidence for improving K1
at 500K. Kaggle1 is the selected desktop platform. Ownership stays on the
dedicated branch through submission, continuation, acceptance and disposition.

## Single intervention

| Arm | Disagreement coefficient | Shared recipe |
|---|---:|---|
| `k1_pretrained_mean2` | 0 | retained 100K stage10 backbone and original seed42 Gap head; two dropout forwards |
| `k1_pretrained_consistency` | 0.1 | exactly the same initialization and two-forward recipe |

Loss is `0.5*(L1(p1,y)+L1(p2,y)) + coefficient*mean((p1-p2)^2)` in normalized
Gap space. Even coefficient0 computes both forwards and disagreement, preserving
RNG consumption and BN updates. No teacher, EMA, architecture or precision change.
K1 atom192/edge64/slot64/layers9, 3,658,817 parameters; AdamW4e-4/wd1e-5/clip1,
unfused/foreachFalse; cosine60 ending1e-6, seed42, FP32/noTF32, BS128/device,
no accumulation. Train-only 500K mean/unbiased std. Epoch permutation seed42+epoch,
drop32 rows/pass. Each arm: 60 completed epochs, 234,360 updates,
29,998,080 presentations. Initial file SHA256
`9656064f7bb0256881da4bc063e90de135171abf517a486bf6202d774694fdba`;
tensor SHA256 `7458f5c4776208da1715d39cbce66537727ff8054b30da0ea07d82d0e94181f4`.
Scientific contract and recipe bytes are generated from the shared owner and
bound to the Spec; no caller override is accepted at runtime.

## Roles and comparison

Accepted `nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1`, manifest SHA256
`630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`.
Pure2D strips packed geometry. Train rows[0,500000), internal development
[500000,550000), already consumed for model selection. Official validation,
test-dev, challenge and common/OOD roles remain sealed. GPU qualification reads
training batches only. Preserve complete role history including checkpoint
initialization lineage and actual packed-target decoding where applicable.

Select each arm independently by best original live development MAE over60
epochs, retaining the full trace and epoch predictions. Primary comparison:
aligned finite 50K selected predictions, delta=consistency MAE minus mean2 MAE;
positive favors removal. Material nomination requires absolute gain>=1meV and
paired-row95% bounds excluding0 in the same direction, 1,000 draws/seed42.
Row intervals do not measure training seed variation. Qualify runtime and all
identity/exposure/artifact fields before calling this a single-intervention
comparison. Historical 500K references are contextual, not silently retrained.

Secondary predeclared analysis applies the accepted `recalibrated_batch_norm`
helper to both selected states: 16,384 unique train rows sampled by
`numpy.random.default_rng(20261008).choice(500000,16384,replace=False)`, in that
draw order, batches128, FP32, only BN modules training, all learned parameters
frozen and dropout off. Evaluate the same full50K development before/after,
retain separate calibrated buffers/predictions, measured native analysis cost
and exact state restoration. Do not reselect epochs or overwrite original
checkpoints. This analysis evaluates BN sensitivity and calibrated endpoints;
its intervals share the existing development-selection limitation.

## Durability and bounded resources

Explicit Kaggle1 T4x2, one arm per device. Both CPU recipe/model/cache checks
and isolated native deterministic optimizer preflights must pass before either
training worker. No local real-model inference is needed for implementation.
Source, configuration, per-arm prospective records and retained initialization
are frozen before submission. Logs report phase and batch/step progress.

Training allocation ceiling:52 T4 device-hours including paired idle capacity;
qualification/installation/CPU analysis are reported separately, with actual
units and unknowns preserved. Historical estimate is21.61 T4 training-hours
per arm; this is not a measured new cost or a one-session guarantee. Each
bootstrap window is at most32,400 seconds. Reuse complete-epoch boundary
checkpointing and immutable retrievable stage manifests. A bounded partial
stage is not terminal scientific acceptance. Continue this same frozen pair
only after authoritative reconciliation, minimal hash-verified artifact
retrieval and cumulative budget review; do not automatically restart or submit
an unrelated successor. A completed peer is retained without retraining.

## Closure

Independently verify each runtime certificate, exact source/data/Spec/job/version,
60-epoch trace/exposure, objective activity, selected predictions/model and
optimizer/RNG/sampler/resume cursor. Reuse existing 500K acceptance and
RML finalization owners. Publish two distinct canonical V5/trace/role/cost
closures; check replay eligibility independently. Missing strict qualification
stays excluded. Write attribution before another module. Adopted positive work
routes to desktop; negative complete histories to archive under BRANCHES.
No full training, official evaluation or model adoption is implied by submission.
