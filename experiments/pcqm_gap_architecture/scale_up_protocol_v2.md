# Resource-Bounded PCQM Scale-Up Protocol V2

## Purpose

This protocol replaces the fixed-batch-across-all-scales assumption for new
promotion ladders. It preserves matched comparisons while allowing a bounded
screen and a full A100 run to use different, predeclared high-batch contracts.
It does not change completed ladders governed by `scale_up_protocol.md`.

## Frozen ladder

Before the first candidate result is visible, freeze
`scale_up_manifest_v2.template.json` with:

1. nested training identities for 50K when used, 100K, 1M, and eligible full;
2. one stage-specific batch, optimizer, warmup, decay, precision, exposure, and
   checkpoint-selection contract;
3. one immutable baseline identity or a fresh matched-baseline requirement at
   every stage;
4. total and per-stage accelerator-hour ceilings.

The inference architecture, graph schema, target, feature implementation,
invalid-row policy, seed policy, and development evaluator stay unchanged.
Only training-row count and the predeclared stage optimization contract may
change between scales.

## Batch ladder

- 50K and 100K: physical batch at least 128 per independently optimized model,
  enforced by `screen_batch_policy.md`.
- 1M and full: physical and effective batch are frozen before S1 from an
  accepted target-hardware capacity probe. They may not be lower than 128.
- A100 should use the largest preflight-accepted batch from the established
  production family, normally 192 or 256 for the present models.
- Comparator and candidate must match exactly within a stage. A score from a
  different batch is not a baseline.
- If device count or gradient accumulation changes, effective batch and
  optimizer-step schedule are recomputed and frozen before training; there is
  no silent fallback after seeing metrics.

## Promotion gates

| Stage | Work | Gate |
|---|---|---|
| S0 | Real-cache forward/backward and durability probe | Finite gradients, physical batch contract, at least 15% device-memory reserve |
| S1a | Optional 50K elimination | Fresh matched candidate/base; both development views improve |
| S1b | 100K seed 42 | Delta at most `-0.002 eV`, paired bootstrap upper bound below zero |
| S1c | 100K seeds 43/44 for one winner | Every seed improves; mean delta at most `-0.001 eV` |
| S2 | 1M bridge | Fresh stage-matched comparison improves by at least `0.001 eV` |
| S3 | One full candidate | Internal views pass under frozen full contract |
| S4 | One official-valid evaluation | At least `0.001 eV` better than a contract-matched baseline |

A failed gate stops the ladder. Official validation stays sealed until S4;
test-dev remains sealed until final inference. No full-data run is authorized
from a low-batch historical score.

## Evidence

Every stage records physical and effective batch, device count, accumulation,
optimizer steps, examples seen, learning-rate trace, throughput, peak memory,
candidate/base aligned predictions, paired deltas, checkpoint hashes, and GPU
hours. Large artifacts stay in platform storage; Git receives compact accepted
evidence and the dated decision.
