# All-step Gap supervision with local atom reconstruction

Prospective authorization: 2026-09-27, one Kaggle2 dual-arm seed-42 study.
This is a training-objective comparison, not an inference-architecture change.

## Evidence and falsifier

The [relation-intervention diagnostic](../pcqm_k1_relation_resolution_100k/diagnostic/decision.md)
found that deleting co-adapted relation branches did not repair transfer. The
[research review](../pcqm_k1_relation_resolution_100k/diagnostic/followup_research.md)
identified a distinct question: local categorical reconstruction while retaining
Gap supervision at EVERY optimizer step. The historical 20-reconstruction +
20-Gap experiment did not test this recipe and used a different contract.

Noisy Nodes (ICLR 2022, section 7.1/Table 7,
https://arxiv.org/pdf/2106.07971) supports joint supervised categorical denoising
on pure-2D PCQM. GPS++ (https://arxiv.org/html/2302.02947, auxiliary-task section)
motivates a small corruption rate; its geometry-assisted system is not evidence
that this isolated 2D recipe will improve K1. No paper's raw-eV coefficient is
silently transplanted into normalized-Gap units.

## Two arms and an existing reference

Both arms instantiate the exact frozen `neural_atom_k1_v4` inference encoder.
No extra slots, geometry, pair branches, teacher, pretraining stage, or seeds.

| Arm | Train input | Objective |
|---|---|---|
| `k1_corrupt_gap` | Select each atom with probability 0.01; replace all nine atom categories with uniformly sampled other valid categories | Normalized Gap L1 |
| `k1_corrupt_gap_atom_aux` | Identical corruption stream | Same Gap L1 + 0.1 x atom CE |

Atom CE is the mean of nine per-feature cross-entropies over selected atoms,
predicting their original categories from final node states. An empty selection
has differentiable zero auxiliary loss. Nine small linear training-only heads
are initialized in an isolated RNG context; inference uses only the unchanged
Gap model. No categorical mask token or vocabulary expansion. Topology, bonds,
RWSE and immutable cached graphs remain unchanged. Perturbed categories are
observation noise, not a claim that chemically modified molecules keep a label.
Development and all terminal inference use clean graphs.

Counter-seeded augmentation is keyed by seed/epoch/batch, identical across the
two arms and isolated from loader, initialization and dropout RNG. Auxiliary
head/optimizer and RNG/counter state are part of atomic continuation checkpoints.
Independent workers share no model, optimizer, RNG, or output directory.

The frozen clean K1 reference is reused, not retrained. A versus K1 measures
corruption regularization; B versus A isolates the joint auxiliary term; B
versus K1 measures the complete recipe. The typed intervention is
`training_objective_comparison`, allowing only `loss_identity` to differ. The
loss identity binds the FULL corruption/loss/head recipe. Clean data features,
inference architecture, optimizer, schedule, target transform and exposure match.
No architecture-only claim is allowed, and equal steps do not mean equal FLOPs.

## Frozen controls, budget and decisions

Exact values and dataset identities live in [training_contract.json](training_contract.json).
BS128/device, FP32 without TF32, seed42, AdamW, original 40-epoch cosine schedule,
drop-last and exact original row order remain fixed. Both T4 workers run once;
six hours maximum including a bounded terminal audit, at most 12 allocated T4
hours plus bootstrap. Expected runtime is an estimate, not a quota reading.
No automatic successor, seed expansion, 500K training, full run or coefficient
sweep is authorized. Infrastructure faults preserve evidence and require
controller reconciliation before any covered repair.

Selection uses minimum clean original-dev Gap MAE, never auxiliary loss. Record
every epoch's Gap/aux/total losses, selected-atom counts, clipping/gradient checks,
LR, actual steps and presentations, checkpoint identity, native wall/device/CPU
cost, runtime/source/data identities and role events. Keep atomic checkpoints
and ten-epoch recovery archives; export clean model, best prediction payload,
final native canonical trace and completion hash manifest independently.

After each arm's complete training artifacts and hashes pass the in-job gate,
reload its frozen best model, reproduce original-dev predictions, then infer
the fixed500K internal development role. This is a separate `NO_TRAIN` action:
no 500K training, weight updates, selection, or checkpoint changes. Save 5K-row
chunks, role/hash/reproduction manifests and stage cost. The role has been reused
in prior discovery and is NOT an untouched final test. Independent downloaded
artifact acceptance is still mandatory. Compare with the retained K1 predictions
on exactly those rows, not a newly trained 500K model.

Report paired row bootstrap on both roles and B-A. A provisional shortlist needs
at least 0.003 eV original-role gain, a paired interval below zero, positive
fixed500K gain of at least 0.001 eV retaining at least half the original gain,
and no material cost/stability failure. These are conservative prospective
decision thresholds, not a claim that run stochasticity was measured. A smaller
gain is `positive_below_gate`, not a promotion. A B win without a B-A advantage
does not attribute improvement to reconstruction. Publish all arms regardless
of outcome; do not choose thresholds after seeing results.

## RML and release

Before submission: actual-reference-bound prelaunch, per-arm prospective RML,
explicit source/config, objective assets, role/trace plans, cost estimate and
one private source publication. No local model execution. Remote preflight
tests categorical bounds, identical initialization/corruption, clean inference,
finite forward/backward/optimizer, auxiliary gradients and checkpoint replay.
Static tests do not substitute for that accelerator preflight.

After completion: independently verify every payload/checkpoint/hash, recompute
three paired contrasts and audit metrics without model inference, bind each
observed comparison and terminal decision, finalize/rebuild/check RML. Training
and audit keep separate identities. Replay readiness is a terminal validator
result, never assumed from prospective files. Historical contracts and production
are unchanged; official validation/test-dev/challenge remain sealed.
