# Frozen K1 fixed-blend transfer qualification

Desktop user authorized this next action on 2026-10-04 ("做吧"). This is
NO_TRAIN clean inference, not a 500K training run or full promotion.

## Question and evidence

The accepted dropout mean2 and consistency2 selected live epoch-37 states
complement each other. The preceding saved-prediction screen measured 4.315135
meV additional gain from an unfitted equal blend on 40K development rows.
This may reflect selection contamination or only that cohort. Freeze both
weights and equal mixing before reading the new role's predictions.

## Inputs and roles

Use OGB PCQM4Mv2 official-training membership only. Reconstruct both accepted
predictions on 2048 evenly spaced source indices in [100000,150000), inclusive
offsets np.linspace(0,49999,2048,dtype=int). Require exact labels and indices,
finite outputs, strict state load and maximum prediction delta <= 1e-4 eV.
Use every internal-development row [500000,550000) for the primary comparison.
These rows were outside both arms' training and checkpoint selection, but were
development for other experiments; they are not a sealed generalization test.
Read no training membership [0,100000), official validation, test-dev or test.
The shard filename's train prefix does not override manifest development role.
Strip all geometric fields through the existing pure-2D packed graph loader.
No geometry construction or optimizer updates.

## Frozen execution and decision

Accepted unchanged K1 family factory, 3658817 parameters, clean eval/no dropout,
FP32, TF32 disabled, deterministic seed42, batch128, fixed normalization
mean5.3383002281188965/std1.275090217590332 eV. Sequential RTX5060 execution.
Allocation budget 600 wall seconds after input hash checking; stop on failed
reconstruction, nonfinite outputs, wrong rows, geometry, or budget expiry.

Compare fixed 50:50 against each constituent on the exact same 50K rows.
Nominate only if gain >=1 meV and paired-row bootstrap 95% lower bound >0
against both; 1000 draws, seed42. The two intervals are exploratory/unadjusted.
No weight fitting or conditional routing. Failure closes this transfer claim;
success supports retained two-model inference and a separate prospective
distillation/scale decision. It does not authorize either training.

Report MAE, aligned artifacts, synchronized forward latency, full-pass wall and
process CPU time, measured assigned RTX device seconds and peak allocated
memory. Separate reconstruction, common-cohort inference and non-GPU overhead.
Single sequential observation per model is not a replicated hardware benchmark.
No new training trace/epochs; historical training costs/replay exclusions remain.
