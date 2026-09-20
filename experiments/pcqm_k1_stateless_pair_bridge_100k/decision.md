# Decision: K1 stateless pair bridge

On 2026-09-20, Kaggle1 version 1 completed the frozen V5 seed-42 screen. Local
acceptance executed no model inference and verified source, immutable data,
runtime, training, artifact, alignment, and sealed-role contracts.

| Model | Parameters | Development MAE | Gain vs K1 |
|---|---:|---:|---:|
| Frozen K1-v4 | 3,658,817 | `0.1413736343 eV` | - |
| K1 stateless pair bridge | 3,689,698 | `0.1411269605 eV` | `+0.0002466738 eV` |

The candidate-minus-reference paired row-error interval was
`[-0.0011152282, +0.0006292391] eV`. It crosses zero, and the point-estimate
gain is only 8.2% of the frozen `0.003 eV` gate. The best checkpoint occurred
at epoch 37 of 40, so extending the same schedule is not justified.

Against the matched recurrent bridge on exactly the same 50,000 rows, the
stateless candidate was better by `0.0005383858 eV`; the direct paired interval
was `[-0.0013944669, +0.0003551091] eV` and crossed zero. It was faster at
834.16 graphs/s versus 765.58 graphs/s and used less peak allocated memory
(608.35 versus 641.52 MiB). This makes recurrence the less efficient option,
but does not establish material value for repeated stateless pair flow.

The V5 outcome is `NEGATIVE_UNDER_CONTRACT`. No retry, width change, extra
seed, 500K bridge, protected-role evaluation, full training, or successor
screen is authorized by this result.

Evidence:

- `results/candidate_acceptance_v1.json`
- `results/recurrent_vs_stateless.json`
- local raw artifact root:
  `platforms/_records/kaggle/training/pcqm_k1_stateless_pair_s42_v1/raw`
