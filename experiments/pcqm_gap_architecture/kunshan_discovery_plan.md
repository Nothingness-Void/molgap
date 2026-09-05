# Kunshan architecture discovery plan

Authorization: 2026-09-05, server-side architecture experiments on Kunshan.
This plan owns hypotheses, ordering, and compute limits. Live job identities
belong in STATUS.md; completed results belong in individual decision records.

## Common comparison

Use the frozen official-PCQM-train-derived 100,000 train / 10,000 internal
validation rows and accepted ETKDGv3/MMFF94s geometry cache. Retain seed 42,
FP32, batch 48, AdamW 1.6e-4, weight decay 1e-6, normalized L1, cosine to
1e-6 over 40 epochs, patience 8, and direct Gap supervision. Train from random
initialization. No official validation/test-dev or other sealed data is used.
The experimental parameter ceiling is 4M. Keep the frozen GraphState9 winner
as the comparison anchor; do not stack unconfirmed candidate mechanisms.

Canonical discovery confirmation runs on SCNet Kunshan, one visible Hygon DCU
per worker. Xi'an may run twice-reproduced paired pre-screens for later
candidates when that materially reduces turnaround, but an Xi'an result alone
cannot promote an architecture. Keep at most one allocated worker per region
and never run the same candidate redundantly across both regions. Platform
Torch/DTK is preserved. CPU-only cache construction and acceptance precede
allocation. The already accepted geometry cache is reused without
reconstruction.

The 3-epoch runtime probe used unnormalized L1 and a constant learning rate.
It is not a matched scientific control and its checkpoint is not resumed.
The first screen therefore trains a fresh complete GraphState9 control with
the original normalized-loss/cosine scientific contract on Kunshan. Subsequent
same-contract candidates may reuse this control only when source, backend,
hardware, input hashes, initialization and shuffle identity remain comparable;
otherwise budget a new matched control. Kaggle results are historical context.

## Experiments

| Order | Question | Concrete change | Role | Estimated DCU-hours |
|---|---|---|---|---:|
| K0 | What is the matched Kunshan control? | Frozen GraphState9, full 40-epoch contract | Required control, not a new architecture | 3.8-5 |
| K1 | Does propagating orientation help beyond scalar bond distances and angles? | A shared, persistent 16-channel order-1 vector state exchanged after blocks 2/4/6/8; real bonds only | First new architecture; paired with K0 | 4.5-6 |
| K2 | Does mean readout discard a useful atom-state distribution? | Replace final mean-only readout with a compact nonlinear projected first/second-moment readout, preserving the encoder | Separate bounded readout experiment | 4-5 |
| K3 | Does conjugated-system communication help beyond local bonds and generic molecule state? | Deterministic conjugated-component membership with a narrow component state; include an information-matched descriptor-only control | Conditional CPU cache plus causal paired screen | 8-12 |

These are planning estimates extrapolated from the 3-epoch probe, not measured
candidate timings. K0+K1 has a 12 DCU-hour hard allocation cap. K2 and K3 are
released only after the preceding result is accepted and attributed by the
coordinator, and after their own protocol and implementation gates pass.
No unimplemented follow-up is silently scheduled as a Slurm dependency.

The user reported 200 available accelerator-hours in each of Kunshan and Xi'an
on 2026-09-06; Slurm does not expose the portal balance and billing conversion
is unverified. Treat this as an inventory ceiling, not a mandate to spend it.
Every candidate remains independently bounded to at most 12 allocated hours,
one seed and one material mechanism. Stop on a platform quota error. Do not
purchase resources or switch to Kaggle/IMS automatically.

## Mechanism justification and exclusions

K1 retains a covariant vector across layers, allowing successive bonded
messages to combine orientation before invariant scalar contraction. The
closed body-order experiment instead injected a fixed invariant summary once
before message passing. K1 creates no contact edges or explicit torsion paths.
It is a small mechanism test inspired by invariant/equivariant coupling in
[GeoMFormer](https://proceedings.mlr.press/v235/chen24ac.html) and the narrow
equivariant designs recorded in the literature ledger, not a reproduction of
their published model or training scores.

K2 asks about the final summary of already contextualized atom states. It uses
no learned queries, attention pooling, extra geometry, or output ensemble; the
failed Query Pool and JK routes are not reopened. The broader motivation is
the GNN readout experiments in
[Molecular set representation learning](https://www.nature.com/articles/s42256-024-00856-0).
Projected moments are a local proposal, not the paper's RepSet implementation.
Its exact width and parameter count must be frozen before model execution.

K3 represents connected conjugated bond systems, which can include acyclic
paths and multiple rings; it does not repeat the closed smallest-ring update.
The [MHNN paper](https://arxiv.org/abs/2312.13136) and its critical follow-up
are indexed in recent_literature_coverage_ledger_50.md. The descriptor-only
control is mandatory: an apparent gain from extra chemical input must not be
misattributed to the communication architecture. Freeze component extraction,
feature availability, and the causal comparison before CPU construction.

## Advancement and stopping

- Accept complete manifests, checkpoints, finite metrics, aligned prediction
  rows/targets, source/cache SHA-256 and the unchanged training contract first.
- Report paired internal-validation MAE difference, parameter count, measured
  throughput and memory. A reduction below 0.001 eV is weak seed-42 evidence;
  it does not justify more seeds under the available budget.
- A promising candidate requires at least 0.001 eV lower MAE, at most 1.5x the
  control's same-device epoch time and at least 15% memory reserve. Preserve
  smaller gains as observations rather than promoting them.
- Seed 43/44, full-data training, official evaluation and production promotion
  remain separate budget decisions. The desktop A100 handoff stays available.
- Failed mechanisms are attributed and closed; no width/seed/schedule grids.
- Infrastructure retries must preserve the contract and consume the same cap.
  Save all RNG/optimizer/scheduler state for an explicit same-run continuation.
- Luna Max collects mechanical terminal evidence in one persistent monitor
  task. It wakes the coordinator once and pauses its heartbeat; only the
  coordinator can select and submit the next frozen experiment.
