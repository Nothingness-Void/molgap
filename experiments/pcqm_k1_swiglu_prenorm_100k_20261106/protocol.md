# Input-normalized SwiGLU100K — prospective protocol

User approved this minimal stabilization and Kaggle3 submission on2026-11-06 JST.
This run stays on the existing SwiGLU question's owner branch. It starts from
the original frozen random K1 initialization, not the interrupted trained state.

## Evidence and falsifier

RML lookup for SwiGLU returns the original INCONCLUSIVE attempt, not a negative
accuracy result. Its actual checkpoint diagnosis is retained in
../pcqm_k1_swiglu_100k_20261106/failure_diagnosis.md and the linked D: reports.
Finite weights and valid development rows produced increasing eval activations
and a NaN; a diagnostic batch-statistics intervention restored finite output.
CPU Torch2.7.1 differs from native2.11.0, so the diagnosis motivates a new
stabilization test rather than proving an exact native failing operand.

Hypothesis: bounding FFN input before multiplicative gating reduces the observed
amplification, allowing a finite40-pass run and a meaningful accuracy comparison.
The cheapest falsifier is the existing sandbox followed by one bounded candidate.
No new baseline training, optimizer sweep or added architecture is authorized.

## Only scientific delta

The new addon `k1_swiglu_prenorm_ffn/1` reuses the original SwiGLU FFN class.
Each of nine FFNs applies `LayerNorm(192, eps=1e-5, elementwise_affine=False)`
before its packed gate/value projection. This adds no learned parameters and
does not normalize the residual branch. Matrix dimensions, dropout, original BN,
slot mixers, head, initialization seed and target transform stay unchanged.
The previous SwiGLU addon and frozen source remain unchanged and recoverable.

Reuse the exact retained reference recipe:100K training/50K development,
40passes,31240updates,3998720presentations, physicalBS128/drop_last,
seed42/python-epoch-shuffle, FP32/noTF32, native Torch2.11 AdamW,
LR4e-4/WD1e-5/clip1, cosine40 to1e-6 without warmup. Select best-development-Live;
track EMA.999 without Clean-BN selection. Keep checkpoint cadence128 and epoch
logging through the existing family trainer. No evaluation batch-statistics
substitution, NaN filtering or guard relaxation.

## Comparator, roles and cost

Reuse the strict original K1 reference
`ev-terminal-T-pcqm-k1-message-droppath-100k-s42-baseline`,
selected Live MAE0.1412944608205557eV. Its exact bundle and recipe are referenced
by experiment.json; prior qualification is documented in
../pcqm_k1_swiglu_100k_20261106/qualification.md.
The interrupted unnormalized SwiGLU is failure evidence, not a complete control
for isolating LayerNorm's accuracy effect. This run tests the stabilized package
against original K1, and separately whether the observed failure recurs.

Use only frozen100K train and50K internal development roles. Official validation,
test-dev and test-challenge remain unauthorized. One candidate uses GPU0 within
the platform T4x2 allocation; record the idle second device in allocation cost.
Budget remains4h allocation wall/8T4-device-hours,180s cleanup reserve.

## Acceptance and disposition

Preserve immutable source, recipe, data, actual remote ID/version, native runtime
proof, full trace/exposure, selected model, aligned finite50K predictions,
checkpoint/resume state, actual cost and observed role provenance through the
existing submission/retrieval/strict-acceptance/terminal RML owners.
The standard numerical guard stops any nonfinite result; do not rescue metrics.

Primary accuracy gate: candidate-minus-reference selected Live MAE<=-3meV,
paired2000-bootstrap95% interval upper<0, and strict comparison qualification.
Require<=2x original parameter count; report actual native cost separately.
Row intervals do not measure training stochasticity. A stable completion alone
does not establish an accuracy gain. Failure remains truthful INCONCLUSIVE;
a complete material-gate miss closes without adoption. No automatic retry,
scale-up or production promotion.
