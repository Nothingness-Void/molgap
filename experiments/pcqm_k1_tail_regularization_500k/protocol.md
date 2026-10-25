# K1 tail regularization500K protocol

## Question and existing evidence

Does stronger late AdamW decay improve generalization from the same trained
K1 predictor? The accepted [fixed-cohort diagnostic](../pcqm_k1_clean_fit_generalization_500k/terminal_decision.md)
reports consistency pass49-to60 TRAIN MAE improvement0.538meV while development
worsens0.787meV. This supports checking late regularization, not a proven unique
overfitting cause, capacity limit or promise of improvement.
The historical60-pass pair kept learning through pass40; do not label all500K
learning stopped in the first10 epochs.

Earlier100K norm/WD work has no terminal result in the reviewed local records;
do not call it negative. This question is a trained500K20-pass tail, not that
from-scratch100K160-pass comparison. Late weight averaging and blind slot-width
expansion already failed their questions and are not repeated.

## Frozen design

- Kaggle1 `nothingnessvoid`, one private T4x2 job, one independent arm per GPU.
- Both start the exact same K1 selected LIVE pass49 tensors,3,658,817parameters,
  state SHA `dfe681a620ffd248b584fec44ccc29fdad38654407a44753b7830d4f33ef2e20`.
  `initialization_provenance.json` binds exact parent bytes, transform and lineage.
- Fresh identical AdamW optimizers; this is warm-start fine-tuning, not original
  optimizer continuation or a random-initialized architecture comparison.
- ArmA `tail_wd1e3`: weight decay0.001. ArmB `tail_wd1e2`:0.01.
  Only this value differs within the job. Both are new candidate recipes.
- One forward/update, normalized Gap L1, clip1, batch128, FP32/TF32off, seed42.
  Sampler Torch pass-randperm; same new-pass order in both arms.
- LR0.0001, native cosine20passes to0.00001.78,120new updates,
  9,999,360new gradient sample presentations per arm. Do not include parent or
  native-probe exposure in the new training counters.
- All500K TRAIN rows0:500000; development500000:550000, previously consumed.
  Manifest SHA630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751.
  OGB atom9/bond3/RWSE16 only; no model geometry, teacher or auxiliary objective.
- Primary: best-development LIVE. Raw EMA0.999 is a shadow report each completed
  pass, not an alternative endpoint chosen after results. No Clean-EMA selection.
  Existing logging: one completed-pass line with state labels/MAE/exposure/LR.
- Before any formal update, reproduce the retained initial full50K predictions:
  identical rows/targets, finite outputs, maximum prediction error<=1e-4eV and
  MAE difference<=1e-6eV, with unchanged initial model/buffers. Mismatch blocks.
- Shared native repeat/resume qualification precedes training. Atomic pass
  checkpoints plus the existing two retained resume chunks preserve recovery.
  No recipe/horizon change is called a resume; no implicit retry is authorized.

## Budget and interpretation

Estimated4.5wall hours/arm,9T4device hours pair; uncertainty includes initial
reproduction, native admission, shadow evaluation and IO. Hard allocation cap
21,600s including setup/qualification,180s cleanup reserve,12T4device hours.
Queue, CPU and native device costs stay separate. These are estimates until
actual receipts, not a guarantee of overnight completion or a new early-stop rule.

Within-job A/B is the strict WD contrast. Check aligned row bootstrap; it does
not measure seed variance. A1meV improvement with positive paired95% interval
nominates B versus A for further evaluation, not automatic adoption/full training.
Each selected LIVE predictor must also improve at least1meV versus the verified
initial predictor to nominate a useful tail recipe. Historical parent comparisons
remain conditional on source/runtime equivalence and do not repair its missing
strict reference/native cost history. Selected versus final points must both be
reported to expose transient best-epoch selection.

After artifacts arrive, reuse frozen inference for the SAME uniform16,384TRAIN
cohort at start/selected/final, with identical evaluation/BN policy. This is a
predeclared endpoint follow-up, not a fabricated clean training curve. The
per-pass stochastic training objective is not clean in-sample MAE. Apply the
same frozen TRAIN-only BN calibration as a secondary diagnostic to each state;
never change the primary endpoint after observing results. No five-pass fixed
cohort callback or new diagnostic runner is added to the training loop.

If both recipes fail, close this bounded tail question and lower priority for
adding exposure; do not infer all regularization, K1 capacity or all500K routes
are exhausted. If only buffers improve, attribute the gain to calibration.

## Closure

Use existing per-arm family inspection, same-run strict comparison, role/cost
records, terminal wiring and RML rebuild/check. Require two distinct complete
replay entries with no exclusions before calling the pair replay-ready.
Native/source/input failures are infrastructure outcomes, not negative science.
Keep official validation/test sealed. No successor, adoption, full-data run or
parent-history repair is implied by this authority.

Negative complete histories route to archive; only adopted positive work routes
to desktop. Runtime status checks belong to Luna. The user requested desktop
shutdown after submission and confirmation both arms actually begin training;
offline time is accepted and does not transfer custody to server.
