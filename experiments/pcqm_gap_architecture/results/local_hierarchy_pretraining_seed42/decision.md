# EdgeState Local-Hierarchy Pretraining Seed-42 Decision

## Question

Does 20 epochs of locally attached atom/bond/functional-group reconstruction
followed by 20 epochs of direct Gap fine-tuning improve the validated EdgeState
GPS9 under the same 40-epoch encoder-exposure budget as scratch training?

## Evidence

Kaggle2 T4x2 kernel
`kaseichou/molgap-pcqm-edgestate-local-hierarchy-s42`, version 1, completed in
6,670.51 seconds. Both arms used the same 100,000/10,000 roles, seed-42 initial
encoder hash, 4,771,073-parameter inference model, FP32, batch 48, optimizer,
learning rate, weight decay, and 40-epoch encoder exposure. Official validation,
test-dev, the shadow role, and the molecular-research server were not read.

| Arm | Allocation | Best epoch | Validation Gap MAE |
|---|---:|---:|---:|
| Fresh EdgeState scratch | 40 Gap | 38 | 0.13777319 eV |
| Local hierarchy | 20 pretrain + 20 Gap | 19 | 0.13800819 eV |

Candidate minus scratch was `+0.00023501 eV`; the required improvement was
`-0.003 eV`. The exact schedule therefore failed its nomination gate.

## Attribution

The local objectives were learnable: aggregate pretraining loss fell from
`0.262839` to `0.112020`; atom, bond, and functional-group terms all declined.
The representation also accelerated downstream convergence. After only 20 Gap
epochs the candidate reached `0.13800819 eV`, whereas scratch through epoch 20
had reached only `0.15327244 eV`. Scratch required epoch 38 to overtake it.

The failure is therefore an allocation failure under the fixed compute budget,
not evidence that the labels damage the model. Half of the candidate's encoder
passes received no Gap supervision, and its best fine-tuning checkpoint was the
last available epoch with a still-improving trajectory. The additional
pretraining throughput cost was modest, but its convergence acceleration did
not produce better final accuracy at equal total exposure.

The first mechanical acceptance used a `1e-8 eV` payload-recomputation
tolerance and rejected two values that differed by roughly `2e-8 eV` because
of float accumulation. Raising only that no-model tolerance to `1e-7 eV`
accepted every frozen identity and did not alter a metric, model, payload, or
scientific conclusion.

## Decision

Close the exact `20 pretrain + 20 Gap` schedule. Do not read shadow, add seeds,
scale to 1M/full, or hand it to desktop.

If a separate compute decision authorizes one follow-up, the highest-information
test is a fresh paired `10 pretrain + 30 Gap` allocation under the same total
40-epoch exposure and every other frozen variable. Pretraining loss had already
captured most of its reduction by epoch 10, while the downstream trajectory was
under-trained at epoch 20. This is a new schedule question, not a repair or an
automatic successor.
