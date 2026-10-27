# Terminal scientific decision: GPTrans BondLocal + PairNorm 500K learning rates

Run `pcqm-gptrans-pairnorm-lr-500k-s42`, Kaggle1 kernel `141037251` version 1.
The frozen protocol compares the same architecture with peak learning rates
`1e-3` and `5e-4`; it does not isolate PairNorm against the historical model.

Both arms completed 60 passes, 234,360 optimizer steps and 29,998,080 sample
presentations. Their selected weights are EMA, chosen using the same 50,000-row
internal development cohort. The saved predictions produce:

| Arm | Selected pass | Selected step | Development MAE (eV) | Decision |
| --- | ---: | ---: | ---: | --- |
| `pairnorm_lr1e3` | 52 | 203112 | 0.099987377815 | CLOSED, same-run reference |
| `pairnorm_lr5e4` | 44 | 171864 | 0.100502566586 | NEGATIVE_UNDER_CONTRACT |

Candidate minus reference is +0.000515188770 eV. Its paired-row bootstrap
95% interval is [+0.000056049251, +0.001034131140] eV, using 2,000 resamples
and seed 42. The frozen gate requires a delta at or below -0.003 eV and a
strictly negative upper interval bound; neither condition is met. The lower
learning rate does not qualify for promotion under this contract.

The development cohort was used for model selection. The interval concerns
rows for this seed and these selected models, not variation across training
seeds or independent-test uncertainty. The older BondLocal-only 500K score
is descriptive context, not a qualified reference for attributing the effect
of adding PairNorm. This decision does not authorize adoption, full training,
or use of protected roles. Strict replay and the physical-cost accounting are
separate qualification results owned by their terminal evidence.

The post-hoc saved-artifact diagnosis is retained at
`D:/molgap-artifacts/reconciliation/gptrans-bondlocal-pairnorm-lr-500k/v1/diagnosis.json`.
