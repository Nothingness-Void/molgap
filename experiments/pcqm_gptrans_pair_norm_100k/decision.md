# Decision — 2026-09-18

Both parameter-free normalization placements completed the frozen 60-epoch
V5 screen and passed mechanical acceptance. The comparison used the immutable
GPTrans-T seed-42 reference and aligned predictions for all 50,000 development
rows. No official validation, test-dev, or challenge rows were read.

| Variant | Development MAE | Gain vs reference | Paired 95% interval | Disposition |
|---|---:|---:|---:|---|
| GPTrans-T reference | 0.156627 eV | — | — | frozen comparator |
| `pair_update_norm` | **0.151784 eV** | **0.004843 eV** | [-0.005813, -0.003863] eV | shortlisted |
| `pair_post_norm` | 0.158614 eV | -0.001987 eV | [0.000990, 0.002987] eV | rejected |

`pair_update_norm` passed both preregistered conditions: at least 0.003 eV
gain and a paired candidate-minus-reference bootstrap upper bound below zero.
It also nominally exceeded the earlier server-owned `pair_prenorm` result
(0.153502 eV) by 0.001718 eV under the same 100K/50K identity, although that
cross-experiment difference was not a preregistered paired gate.

The placement result is directional evidence about information flow. Scaling
each newly generated relation update before residual addition helped, while
normalizing the accumulated pair memory after every addition hurt. This
supports bounded update injection while preserving the magnitude and history
stored in persistent pair state. It does not prove that activations had
exploded or establish full-scale superiority.

Both candidates selected epoch 59 and their development curves were still
improving as the cosine schedule reached its fixed endpoint. The result is a
seed-42 mechanism shortlist only. It does not authorize altered schedules,
additional seeds, 500K/full training, official evaluation, or production
promotion without a separately frozen release decision.

Exact no-inference acceptance and artifact identities are recorded in
`acceptance.json`. Execution provenance is under
`platforms/_records/kaggle/training/pcqm_gptrans_pair_norm_v5_s42_v2/`.
