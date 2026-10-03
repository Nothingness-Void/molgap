# Chemical component 100K pair release, 2026-09-30

The user authorized resubmitting GPTrans chemical auxiliary losses on Kaggle1.
This release follows the standing two-new-direction instruction documented in
`submission_attempt_review.md`; the older baseline-plus-joint draft was not run.

## Arms and fixed recipe

- `descriptor_aux`: normalized Gap L1 plus 0.1 masked descriptor MSE.
- `fingerprint_aux`: normalized Gap L1 plus 0.1 unweighted fingerprint BCE.

Both use the same frozen GPTrans-T seed42 initial backbone, 288->32->712
auxiliary head, clean original graph inputs and independently accepted train
caches. The inactive auxiliary loss is skipped. Head initialization uses an
isolated CPU RNG context. Exported inference keeps the original Gap model.

Reuse the existing V4 trainer: FP32, TF32 disabled, deterministic algorithms,
batch128, drop_last, 781 steps/epoch, 60 epochs, 46,860 optimizer steps and
5,998,080 sample presentations. AdamW lr0.001, weight_decay0.05, clipping1.0,
four-epoch linear warmup then cosine to 0.000001, EMA0.9999 and best-development
EMA selection remain frozen. No early-stop policy, warm start or geometry input
is added. Each arm uses exactly one isolated T4 in the same T4x2 allocation.

Train indices 0..99,999 and development indices 100,000..149,999 use the
accepted V4 graph manifest. Auxiliary labels use train SMILES only under
`label_policy.md`. Official validation, test-dev and test-challenge stay sealed.

## Release and evidence

Full cache acceptance precedes both prospective training plans. Reuse the
experiment CLI for Spec/package/release/real-loader checks and launch receipts.
The existing paired launcher delegates to `run_arm.py`; GPU runtime preflight
and the existing <=5% synchronized optimizer-step overhead gate precede formal
training. Step timing is not end-to-end overhead qualification. A rejected
preflight stops this launch and retains its exact diagnostic output.

Each arm retains independent family-output identity, 60-epoch trace, selected
model, aligned 50K predictions, resume checkpoint, actual native-cost evidence
and role records. Complete output is subject to owning acceptance and RML
terminal validation; submission alone is not replay readiness.

## Comparison and decision

Reuse `pcqm-gptrans-t-100k-v4-reference` and its frozen prediction bundle
(MAE 0.15662720430791377 eV). The intervention deliberately changes the training
objective; retain that difference in scientific and cache identity. Do not
represent it as an unchanged-loss V4 comparison or as a same-job reference.
Validate the applicable V5 comparison qualification at acceptance; a missing
qualification remains pending/inconclusive and cannot be waived by this file.

Evaluate each arm independently and compare the two new mechanisms on aligned
rows. The predeclared nomination threshold is 3 meV improvement and a positive
paired improvement interval against the retained reference, plus the owning
cost qualification. Row bootstrap does not estimate training stochasticity.
No automatic 500K, full-scale or production adoption follows.

Budget is at most six estimated T4 device-hours per arm after runtime preflight,
with concurrent packing fitting Kaggle's time limit. CPU cache cost, queue time,
preflight cost, training wall time and native T4 allocation remain separately
recorded. Unknown costs remain unknown. Checkpoints are atomic each epoch and
outputs remain retrievable after completion or failure.
