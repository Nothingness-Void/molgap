# SignNet-LapPE GraphState seed-42 decision

## Result

The accepted eight-mode normalized-Laplacian cache covered all 100,000 train
and 10,000 internal-validation graphs. Kaggle2 version 1 then completed both
40-epoch T4 workers and passed no-model acceptance.

The SignNet-LapPE candidate reached `0.1305700988 eV`; its fresh GraphState9
control reached `0.1298621446 eV`. The candidate was worse by
`0.0007079542 eV` (0.5452%). It added only 7,300 parameters (+0.20%) and ran at
0.9822 times control throughput, so excessive compute was not the cause.

## Attribution

The candidate's epoch-39 training MAE was lower (`0.1131444464` versus
`0.1142505991 eV`) while its validation MAE was higher. Its validation-minus-
training gap widened by `0.0018141068 eV`, and it beat the control in only four
of the final ten epoch-aligned validation measurements. The zero-initialized
spectral return had finite nonzero gradients, so the branch was connected and
learned; the failure is scientific rather than an inactive implementation.

RWSE16 already summarizes random-walk return probabilities, which are spectral
moments of graph topology. Adding eight low-frequency normalized-Laplacian
modes therefore supplies largely overlapping global topology while exposing
symmetry/eigenspace-specific fitting freedom. Under this direct-Gap 100K
contract, that extra freedom reduced training error but did not generalize.
This result rejects the tested SignNet-LapPE injection, not every possible
spectral method in unrelated data or pretraining regimes.

## Disposition

The mechanism is closed without seed, eigenmode-count, width, optimizer, or
schedule variants. It does not replace the three-seed GraphState9 handoff.
Full-data training, official validation/test-dev, production changes and
molecular-research-server use were not authorized.
