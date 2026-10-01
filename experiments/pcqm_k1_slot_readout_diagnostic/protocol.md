# K1 slot/readout diagnostic v1

Desktop-owned observational diagnostic of retained width192/256 checkpoints.
The accepted width256 bounded negative result remains immutable.
No training, optimizer, parameter update, geometry construction, or official validation/test.

## Question and falsifier

Does final K1 slot return become weak after mean pooling, and do width256
regressions concentrate by 2D graph size or cyclic/aromatic/conjugated structure?
The eval identity mean(delta_h)=u/N does not alone establish harmful attenuation:
u may adapt with N and learned head weights may compensate.
Measure norm ratios, assignment entropy/effective atom count, and size associations.
Width remains a cross-checkpoint observation with single-seed/runtime confounding.
No ablation or output-rescaling intervention is released by this contract.

## Inputs and roles

Hash-pinned selected_model, saved development predictions, source archive/extracted
Python files, and retained cache development shard in inputs.json.
Only source_idx [100000,150000), already consumed for selection. Strip geometry
through the existing pure2D loader; only original atom9/bond3/RWSE16 enter models.
No training-role rows loaded; frozen denormalization mean/std from accepted contract.
Checkpoint state strict loading and parameter counts 3658817/6035201 required.

## Observations

Uniform sample of 2048 offsets: numpy.default_rng(20261002).choice(50000,2048,
replace=False), sorted. CPU FP32 eval/inference_mode, four threads, batch64,
workers0. Read-only hooks at mixers3/6/9; no altered forward output.
Max CPU-vs-retained prediction difference must be <=1e-4 eV or model
interpretation stops. Measure pre-mixer atom-mean norm, mean-update norm,
their ratio, sum-update norm, entropy, effective atom count, max assignment,
and numerical single-slot mean-return identity. Report pergraph values.

Full50K graph metadata joins exact saved rows and targets. Size bins <=15,
16-25,26-35,>=36; cycle rank E-N+components bins0,1,2,>=3. OGB boolean aromatic,
in-ring and conjugated fractions only with verified feature mapping. Fraction
quartiles are descriptive posthoc boundaries. Paired row bootstrap1000 seeded
replicates; row uncertainty is not training stochasticity. No causal cohort claim.

## Cost and stop

Each worker wall ceiling600seconds; total scientific worker ceiling1200seconds.
Record process CPU and wall separately; local GPU/device and queue not applicable.
Atomic outputs per arm retain progress. Stop on changed hashes, missing/unaligned
rows, nonfinite results, reconstruction failure or ceiling; no retries/training.
Planning/implementation/git wall outside measured worker windows remains unknown.

## Interpretation

CONTEXT_ONLY; no performance promotion, strict causal comparison, replay-ready
trace, new protected role or automatic successor. A large learned slot contribution
falsifies the simple claim that it is numerically negligible. Small contribution
supports an intervention question but cannot prove that its information is useless.
Preserve terminal failure attribution and distinguish mechanism algebra from effect.
