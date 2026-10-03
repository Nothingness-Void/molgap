# Local RML census and FLAG route review — 2026-10-02

## Scope and evidence status

Read-only local census across the desktop checkout, indexed local Git worktrees, the existing cached origin/molgap-server Git object, older local molgap-server branch, prior review inventory, and archive tip. No fetch, remote/platform call, model execution or training was performed; no credentials or protected data were read. No generated RML files were changed or rebuilt. No tests were run.

- Current desktop RML index: 67 trajectories at HEAD e23e1a41280a8441942ce16b6ed4e8d4b16e641f; source digest 4a796c19b6b7838839a2a0eb43b7c3326ddfa3f9043a95ee85434e37115495ce; index SHA256 6e6dddfade92510da397fc0639f5b5bc8bb310280ad0b3e5237993673a719b7c.
- Indexed worktree census: 63 local worktrees, 30 with RML indexes, 1,078 indexed rows and 103 unique IDs. Repeated clones are memberships, not new trajectories.
- Latest cached origin/molgap-server index: 64 trajectories at cac66f69166260254ae5cb3f69433d181c06592c; source digest 5dc035bcea2b7e1edd74ec457cb9da6a88832d3f1444eefa1c1baef4a60f9d6c. Older local molgap-server has 25 entries, all represented in the cached latest snapshot. No fetch/network call was made.
- Combined indexed worktree + cached server union: 166 unique IDs. Prior 2026-09-30 navigation inventory: 126 IDs across 240 rows; its IDs were covered by the current union. That file is navigation only, not evidence reacceptance.
- Archive tip f16823acc012727c02671b64798147346ba70010 has no derived trajectory index and contains eight trajectory JSON snapshots representing four IDs, all overlapping indexed IDs. Relevant archived decisions are cited below. This census does not claim that every unindexed historical blob at every archive ancestor was read.

Canonical decisions outrank generated indexes. Owner-local evidence is preserved with its snapshot; an indexed ACTIVE label is not scheduler truth. Do not reinterpret server-owned records as desktop validation or take over server jobs.

Desktop derived ledgers report 8 complete native cost measurements, 48 incomplete, and 11 with no cost event; 98 cost events include 17 estimates, 82 measurement-missing, and 40 not-applicable entries. Cross-hardware total is null. The role index has 126 explicit events, 244 ambiguous historical records, and three full-training membership conflicts. Cost units remain separate, unknown is not zero, and many historical RML records are partial or lack traces/V5 evidence.

## Candidate assessment

The indexed local canonical record/decision/attribution search found no executed FLAG, gradient-adversarial embedding, or SAM trajectory. This is a bounded finding over enumerated local snapshots, not an absence claim about unindexed files, every archive ancestor, or external literature.

The frozen candidate is supervised gradient-directed perturbation at the categorical atom embedding in pure-2D K1/EdgeState Gap regression. It is genuinely untested in searched local RML, not a predicted breakthrough. Capacity expansion and nearby auxiliary routes missed the material gate; this intervention tests an unmeasured training constraint while keeping the clean inference graph unchanged. The protocol fixes 3,658,817 stored/trainable parameters, zero added inference parameters, below the 7,317,634 two-times ceiling. Three perturbed loss evaluations per batch are three times the gradient-bearing row evaluations, not equal-compute evidence. The existing protocol bounds one 100K/40-epoch candidate and consumed 50K development selection; it does not release official validation/test roles or 500K/full training. See protocol.md, role_plan.json, and evidence_review.md.

## Mechanism and outcome synthesis

- Input-category repair resolved graph collisions and is already incorporated; preserve it and persistent real-bond EdgeState. See experiments/pcqm_edge_state_full/root_cause.md.
- Capacity: K1 atom width 192→256 is negative, about 1.504 meV worse at retained seed42/40 epochs. Slot width 64→96 gains about 0.547 meV, below the 3 meV gate with paired interval crossing zero. Final slot usefulness does not establish latent saturation. Owners: D:/w/k1-width256/experiments/pcqm_k1_node_width256/terminal_decision.md; D:/w/k1-slot96/experiments/pcqm_k1_slot_width96/terminal_decision.md and attribution.md; D:/w/k1-slot-diagnostic/experiments/pcqm_k1_slot_readout_diagnostic/terminal_decision.md and attribution.md.
- Attention/value/routing: local denser attention is unsupported. Desktop PairToken value decoupling is inconclusive under strict qualification (2.411 meV point gain, below 3 meV; reference/replay proof missing). Server PairToken narrowly clears the 100K gate but fails transfer at 500K. RRWP, receiver-pair, triplet aggregation, SPD bias, and token variants are mostly positive-below-gate. Server results remain independently owned.
- Nearby negative outcomes include multiplicative pair value, sparse triplet, linear attention, persistent triplet, motif hierarchy, hidden BN/context gate, MoSE replacement, and local/archive query pooling (6.936 meV worse). Corrupted-gap/atom auxiliary and MoSE residual are below-gate positives; reconstruction/corruption pretraining has not established portable superiority over clean K1. These do not test supervised embedding adversariality. Label-informed Oracle is an upper bound; soft saved-prediction routing missed its 1 meV gate and is NO_TRAIN.
- Archived GraphState allocation reported about 6.752 meV mean gain with fewer parameters under a different 100K/10K internal split; it is not a matched current K1 comparator. Pointers: archive:f16823acc012727c02671b64798147346ba70010:experiments/pcqm_gap_architecture/results/local_global_allocation_seed42/decision.md and local_global_allocation_multiseed/decision.md; query_pool_seed42/decision.md; post_dual_stream_failure_attribution.md.
- Owner update: SSMA was accepted as terminal negative at D:/w/k1-local-aux/experiments/pcqm_k1_local_mixing_clean_aux/kaggle3_accuracy_reconciliation_v1/terminal_decision.md, attribution.md, and scientific_metrics.json. Reference MAE 0.1412944608205557 eV; SSMA 0.14336151182115078 eV; gain −2.067051 meV; 10,000-draw interval [−3.005820, −1.105339] meV; overhead 48.977%. This closes the dependency and does not imply an SSMA rerun or a FLAG result.

## Limits

The detailed authoritative artifacts remain beside their owning experiments. The companion rml_inventory.json records the bounded counts, primary snapshot locators, exact method-search regex, hashes, and provenance boundary. It is intentionally a review inventory, not canonical RML, V5 acceptance, scheduler state, Kaggle acceptance, or validation against an unaccepted owner branch. Server evidence remains read-only.

## Reconciliation update before prospective publication

SSMA accepted closure was imported on desktop dcdda03518a281c2b06a92e374991883f4f737d4, adding frozen reference/NO_TRAIN ancestry as required. Desktop validation, generated rebuild and frozen check pass with74 trajectories,63 V5 evidence,113 cost events,145 explicit roles and21 traces; READY0. The cost-completeness summary above is the earlier67-trajectory snapshot, not a refreshed complete-cost census. Complete negative SSMA owner d92f361 is durably reachable through origin/archive6b91b2857e0eb014f8c3ab6376638280ceffa9fe. The reviewed archive f16823 snapshot remains pinned historical evidence. Desktop portable verification retains the documented missing historical reference raw trace; no replacement trace was fabricated.
