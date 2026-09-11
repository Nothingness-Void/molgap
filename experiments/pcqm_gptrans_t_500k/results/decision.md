# GPTrans-T core 500K decision

Decision date: 2026-09-11

## Result

The SCNet Kunshan run completed all 60 direct-Gap epochs on the accepted
500K/50K cache. Its downloaded best-development payload contains exactly
50,000 finite predictions with ordered source indices `500000..549999`;
locally recomputed MAE agrees with the remote metric.

| Model | Parameters | Best epoch | Development MAE |
|---|---:|---:|---:|
| Matched ESGPS6-304 scratch | 7,610,945 | 48 | 0.10767170 eV |
| GPTrans-T propagation core | 5,246,817 | 52 | **0.10394785 eV** |

GPTrans-T improves MAE by `0.00372385 eV`, exceeding the frozen `0.001 eV`
gate while using about 31% fewer inference parameters. The propagation-core
transfer is accepted as a positive 500K architecture result.

## Boundary

This implementation intentionally omits the paper's offline multi-hop
edge-sequence tensor and therefore is not a byte-identical reproduction. The
result nominates this adapted core for a separately budgeted confirmation or
scale decision; it does not authorize official validation/test-dev access or
an automatic full-data run. The concurrently running GPS9 distance-angle pair
has a different optimizer and must not be treated as a matched comparator.

Mechanical evidence is in [`acceptance.json`](acceptance.json). Remote job and
cache identities remain in [`../remote_launch.json`](../remote_launch.json).
