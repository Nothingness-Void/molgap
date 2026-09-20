# Decision: K1 recurrent pair bridge

On 2026-09-20, Kaggle1 version 1 completed the frozen V5 seed-42 screen. Local
acceptance executed no model inference and verified source, immutable data,
runtime, training, artifact, alignment, and sealed-role contracts.

| Model | Parameters | Development MAE | Gain vs K1 |
|---|---:|---:|---:|
| Frozen K1-v4 | 3,658,817 | `0.1413736343 eV` | - |
| K1 recurrent pair bridge | 3,689,698 | `0.1416653395 eV` | `-0.0002917051 eV` |

The candidate-minus-reference paired row-error interval was
`[-0.0006022758, +0.0011791791] eV`. It crosses zero and the point estimate is
a regression. The best checkpoint occurred at epoch 35 of 40, so the failure
is not evidence for extending the same schedule.

The persistent 32-channel ordered-pair state at layers 3, 6, and 9 added
30,881 parameters. It trained at 765.58 graphs/s, with 641.52 MiB peak
allocated memory. The matched stateless bridge was faster and had a lower
point-estimate MAE, but their direct paired interval also crossed zero. The
available evidence therefore supports neither persistent pair memory nor a
larger version of this mechanism.

The V5 outcome is `NEGATIVE_UNDER_CONTRACT`. No retry, width change, extra
seed, 500K bridge, protected-role evaluation, full training, or successor
screen is authorized by this result.

Evidence:

- `results/candidate_acceptance_v1.json`
- local raw artifact root:
  `platforms/_records/kaggle/training/pcqm_k1_recurrent_pair_s42_v1/raw`
