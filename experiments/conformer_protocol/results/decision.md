# ETKDG versus ETKDGv3+MMFF 50K Decision

## Decision

Keep `ETKDGv3 + MMFF(maxIters=200)` for Route B construction and inference.
Reject bare ETKDG as the default acceleration path.

On this machine, MMFF adds only `161.96 s` for 50K molecules while improving
the frozen four-encoder Route B equal-seed ensemble average/Gap MAE by
`0.009868/0.008612 eV`. This is a favorable one-time CPU-cache tradeoff.

## Fixed Protocol

- Rows: 40K train, 5K validation, and 5K test selected from the existing
  scaffold-disjoint PubChemQC 100K split.
- Speed: all 50K rows, 12 local workers, 1K-row atomic shards.
- Accuracy: the untouched 5K test rows only.
- Controlled variable: the same molecule, ETKDGv3 seed, labels, checkpoints,
  and heads; only `MMFF(maxIters=200)` is enabled or disabled.
- Frozen architecture: GPS9-192 + GPS11-160 + primary SchNet-176/160/6 +
  two-conformer-trained SchNet-176/160/6 evaluated on the same primary view +
  `+-0.10 eV` bounded-residual head.
- Head stability: seeds 42, 43, and 44; the table reports their prediction
  ensemble. Every individual head seed has the same conclusion.

This is an inference-time geometry A/B with frozen checkpoints. It does not
measure the ceiling from retraining every encoder and head on bare ETKDG.

## Results

| Geometry | 50K wall time | Throughput | Average MAE | Gap MAE |
|---|---:|---:|---:|---:|
| Bare ETKDGv3 | 275.08 s | 181.76 molecules/s | 0.157571 eV | 0.188062 eV |
| ETKDGv3 + MMFF200 | 437.04 s | 114.40 molecules/s | 0.147703 eV | 0.179450 eV |
| Accepted current MMFF cache | reused | reused | 0.146881 eV | 0.178796 eV |

MMFF is `1.589x` the bare construction wall time and costs an extra
`2.70 min/50K`, `5.40 min/100K`, or `0.90 h/1M` at the measured 12-worker
throughput. The paired average-MAE improvement is `6.26%`; its normal 95%
confidence interval is `0.008530-0.011207 eV`. The thresholded fraction of
test molecules that improve by more than `0.0001 eV` is `58.20%`, versus
`39.36%` that degrade by more than that threshold.

The 3D encoders show a larger direct effect:

| Frozen 3D encoder | Bare average MAE | MMFF average MAE | Improvement |
|---|---:|---:|---:|
| Primary SchNet | 0.231400 eV | 0.180375 eV | 0.051024 eV |
| Two-conformer-trained SchNet | 0.219976 eV | 0.170100 eV | 0.049876 eV |

The bounded-residual head intentionally limits how much of that 3D gain reaches
the final prediction. The existing accepted MMFF cache reproduces the paired
MMFF result within `0.000822 eV`, which rules out a material evaluation-path
or deterministic-seed artifact.

## Construction Notes

Both protocols produced 49,998 of 50,000 graphs; the same two training rows
failed ETKDG embedding, so the 5K test comparison remains fully paired. Of the
49,998 MMFF calls, 15,632 converged within 200 iterations, 33,509 reached the
iteration limit, and 857 returned an MMFF setup failure. The current graph
contract retains the generated coordinates in all three cases. Despite the
large non-converged fraction, the partial optimization is materially better
than no optimization.

## Evidence

- Selection and identity contract: `selection.json`
- Regenerable graph shards and construction summaries:
  `data/cache/phase8/conformer_protocol_50k/`
- Frozen-model metrics and paired statistics: `evaluation.json`
- Cost/accuracy summary: `tradeoff.json`
- Reusable implementation: `src/molgap/conformer_ab.py`
- Thin entrypoint: `scripts/phase8/experiments/run_conformer_ab.py`

No sealed rows were used, no model was retrained, and the production registry
was not changed.
