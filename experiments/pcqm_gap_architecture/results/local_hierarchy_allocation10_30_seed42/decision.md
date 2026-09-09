# EdgeState Local-Hierarchy 10/30 Allocation Seed-42 Decision

## Question

Does reallocating the fixed 40-epoch encoder-exposure budget from 20/20 to 10
local-hierarchy pretraining epochs and 30 direct-Gap epochs produce a material
improvement over fresh EdgeState scratch training?

## Evidence

Kaggle2 T4x2 kernel
`kaseichou/molgap-pcqm-edgestate-hierarchy-10-30-s42`, version 1, completed in
6,290.46 seconds and passed independent no-model acceptance. Both arms used the
same 100,000/10,000 roles, seed-42 initial encoder hash, 4,771,073-parameter
EdgeState GPS9 inference model, FP32, batch 48, optimizer, learning rate, weight
decay, evaluator, caches, and 40 encoder passes over the train role. Official
validation, test-dev, shadow, and the molecular-research server were not read.

| Arm | Allocation | Best epoch | Validation Gap MAE |
|---|---:|---:|---:|
| Fresh EdgeState scratch | 40 Gap | 39 | 0.13749303 eV |
| Local hierarchy | 10 pretrain + 30 Gap | 29 | 0.13515098 eV |

Candidate minus scratch was `-0.00234205 eV`. This is a clear paired positive
signal, but it falls `0.00065795 eV` short of the predeclared `0.003 eV`
nomination threshold. Mechanical acceptance passed; the scientific promotion
gate did not.

## Attribution

The allocation change converted the prior 20/20 result from `+0.00023501 eV`
regression into a `-0.00234205 eV` gain. The local objectives were learned
quickly: aggregate pretraining loss fell from `0.262839` to `0.122825` in ten
epochs, while atom, bond, and functional-group terms all declined. Giving 30
rather than 20 epochs to direct Gap supervision therefore retained useful
initialization while allowing substantially more target adaptation.

The effect is larger than the `0.00028016 eV` difference between the fresh
scratch controls in the 20/20 and 10/30 jobs, which supports a real signal.
However, both paired arms achieved their best checkpoint at their final epoch.
The candidate may benefit from more Gap exposure, but granting it more than 40
total encoder passes would break the equal-compute question. Trying additional
allocations on the same selection role would also turn a bounded hypothesis
test into validation-set schedule search.

Pretraining averaged 621.50 graphs/s versus 655.10 graphs/s for scratch Gap
training; candidate Gap fine-tuning averaged 664.57 graphs/s. The mechanism is
therefore affordable and useful as a convergence prior, but its measured gain
is below the deliberately conservative threshold for spending shadow, extra-
seed, or scale-up budget.

The concurrently adopted batch-128 screen policy does not promote this result.
It requires an older batch-48 positive to receive a fresh batch-128 paired
confirmation **before promotion**; this experiment did not pass its own frozen
nomination gate, so no confirmation job is released.

## Decision

Record the exact 10/30 schedule as **strong positive but not nominated**. Close
the local-hierarchy allocation question under its frozen gate. Do not test
another allocation, read shadow, add seeds, scale to 1M/full, hand it to
desktop, access official roles, or use the molecular-research server.

The evidence supports local-hierarchy reconstruction as a useful transferable
idea, not as an authorized Track B winner. Reopening it would require a new
question, a fresh audit role, and an explicit compute decision rather than
continued tuning on this validation role.
