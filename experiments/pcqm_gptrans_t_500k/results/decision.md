# GPTrans-T core 500K decision — 2026-09-11

The SCNet Kunshan run completed all 60 direct-Gap epochs on the accepted
500K/50K cache. Its 50,000 aligned development predictions passed local
no-inference acceptance.

| Model | Parameters | Best epoch | Development MAE |
|---|---:|---:|---:|
| matched ESGPS6-304 scratch | 7,610,945 | 48 | 0.10767170 eV |
| GPTrans-T propagation core | 5,246,817 | 52 | **0.10394785 eV** |

GPTrans-T improved by `0.00372385 eV`, exceeding the frozen `0.001 eV` gate,
while using about 31% fewer inference parameters. It is accepted as a positive
500K architecture result.

This implementation omits the paper's offline multi-hop edge-sequence tensor
and is not a byte-identical reproduction. The result nominates the adapted core
for a separately controlled scale decision; it does not authorize official
validation/test access by itself.

Machine-readable evidence: `acceptance.json`.

