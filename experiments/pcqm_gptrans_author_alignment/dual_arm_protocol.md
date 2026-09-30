# GPTrans input-attribution dual arm

Frozen design date: 2026-09-30. Owner: server. User authorization: one G1/G2
dual-arm experiment and measurement of submission preparation latency.

## Question

Can input scale or omitted chemical path information explain part of the weak
adapted GPTrans endpoint without changing the propagation core or exposure?
Evidence: [real-input diagnostic](decision.md), [accepted complete V5 reference](../pcqm_gptrans_v5_audit_reference/results/terminal/decision.md).
This is not an exact reproduction of the published model or its training budget.

## Independent interventions

- **G1 degree scale:** multiply only the initial in/out degree tables by 0.0897.
  Preserve every other initial tensor. This rounded factor is motivated by the
  measured atom/degree energy ratio, not selected with development labels.
- **G2 path bond mean:** preserve the complete initial state and every original
  parameter. Add the mean of the existing categorical bond embeddings along
  one deterministic shortest path to nonbonded pairs at distances 2 through 20.
  Direct bonds, spatial encodings and propagation remain unchanged. Sorted
  neighbor BFS resolves ties. Diagonal, unreachable and over-cap pairs get no
  added encoding. This deliberately tests chemical path content without the
  author's additional learned hop-position transforms.

G1 and G2 are independent candidates against the same immutable comparator,
not a combined model or factorial interaction estimate. The alternative
explanations are LayerNorm washing out the input scale, and path pooling losing
important sequence order; a negative result cannot rule out all author encodings.

## Scientific constants and evaluation

Reuse the complete [reference contract](../pcqm_gptrans_v5_audit_reference/contract.json):
fixed 100K training / 50K internal-development rows, seed42, FP32/no TF32,
physical BS128 per model, drop_last, 60 epochs, four warmup epochs, unchanged
AdamW/loss/schedule/EMA selection and exposure. No official validation,
test-dev/challenge, other seeds, automatic 500K/full training or prediction fusion.
The accepted V5 reference is reused, never redundantly trained.
The original 0.003 eV material gate is retained; it is not a measured seed variance.

## Dependency and compute order

1. Tensor-only G1 preparation and label-free G2 sidecar construction on CPU.
   Inputs are the accepted immutable graph shards and the frozen initial state.
   Do not read y/pos for path construction; derive only from x/edge_index/edge_attr.
2. Independently verify complete row order, path policy/content, original shard
   hashes, initial-state transformation and output hashes. CPU outputs are
   retrievable per original shard; never build paths during GPU allocation.
3. Freeze candidate recipes, Spec, package, per-arm prospective RML and actual
   comparison prelaunch. Extend the owning trainer's complete live/EMA recording
   explicitly for these arms; never impersonate the reference run.
4. `check-release` binds source dependencies, LF recipe bytes, frozen states,
   actual entry script and staged inputs. The owning Kaggle adapter rechecks
   the report before POST and retains the actual platform response.
5. One T4x2 job, independent RNG/model/optimizer/checkpoints/trace per arm;
   each worker sees one GPU. Optimizer-inclusive preflight must pass before its
   training loop. Checkpoint atomically every epoch and retain bounded chunks.
6. Saved-artifact acceptance, aligned paired analysis, costs/roles, separate
   terminal decisions and replay-ready RML publication. Submission/COMPLETE
   alone cannot establish scientific acceptance or replay readiness.

GPU release stays blocked until steps 1 through 4 pass. A source-package check
cannot replace complete cache acceptance or a compatible executable trainer.

## Timing and resource accounting

Record two different clocks: request-to-first-remote-action and
fully-qualified-package-to-submission. The five-minute target concerns process
reuse; do not hide new scientific engineering, CPU dependency completion,
upload/queue duration or failed attempts in that measurement. A missed target
is reported as missed, not repaired by skipping gates.
CPU preparation ceiling: three wall hours, no GPU. Planned dual training ceiling:
six wall hours / twelve allocated T4-device-hours including preflight. Actual
reservation is frozen again from accepted CPU outputs and measured reference
runtime before GPU release. Unknown submit outcomes require reconciliation,
not an automatic duplicate. The existing server B monitors the exact bound job;
only A can accept it and release the next covered stage.
