# Prospective protocol — independent 2D multi-hop MetaGIN-family backbone

## Question and evidence

Does replacing K1's local EdgeState plus molecular-slot exchange with a
four-stage, sequential real-bond/2-hop/3-hop message backbone and persistent
virtual molecular state improve direct PCQM Gap prediction under the *same*
accepted fixed-100K V5 scientific contract? The mechanism changes the
information flow, not merely the width of a K1 adapter. The motivating
[route review](../pcqm_v5_route_portfolio/evidence_review_2026-09-28.md)
closed repeated K1 local-capacity additions. The [MetaGIN paper](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-024-3784-y)
is an architecture prior, not performance evidence under this contract.

This is an independently written **adaptation**, not an exact reproduction of
the authors' SDF preprocessing, Adan/BS256/144-epoch recipe, implementation,
or published score. Its only raw inputs are the existing accepted OGB nine
atom categories, three real-bond categories and RWSE16. Directed simple-path
multiplicity for hops 2 and 3 is deterministically derived from those real
bonds by a CPU-only sidecar; no coordinates, angles, torsions, teacher,
pretraining, extra labels, all-pairs attention, or official evaluation role.
The sidecar is *derived topology*, not a replacement graph dataset.

## Frozen falsifier

One MetaGIN2D width-256/depth-4/hop-3 candidate, seed 42, starts from random
initialization. It is compared with the immutable [K1-v4 reference bundle](../v5_legacy_evidence_migration/k1_v4_100k_reference/reference_bundle.json);
K1 is not retrained. The [contract](training_contract.json) freezes physical
BS128, FP32/TF32-off, source-index roles/order, normalized direct-Gap L1,
AdamW, cosine schedule, 40 epochs and 3,998,720 presentations. Only the
architecture config changes; platform and accelerator remain provenance. The
50K internal-development role is reused for selection, not an independent
holdout. Official-valid/test-dev/challenge remain sealed.

1. Static and synthetic CPU tests must establish exact simple-path counting,
   node permutation invariance, disconnected/isolated behavior, batching,
   finite gradient and parameter identity.
2. A separate CPU-only Kaggle job must derive all 150,000 rows from the
   accepted fixed-cache shards, record SHA-256 per shard and fully recompute
   each row for independent acceptance. The GPU job can mount only this
   accepted sidecar plus the unchanged fixed cache.
3. GPU runtime preflight must hash the mounted source and target-transform
   asset, verify all fixed source indices/label bytes and sidecar hashes,
   observe one cold CUDA step plus two identical warmed FP32 optimizer steps,
   retain at least 15% VRAM, and
   reject estimated >6-hour worker time. Epoch 0 measures the full train+dev
   pass and fails early if 40 epochs project beyond that budget.
4. If admitted, every epoch emits an atomic model/optimizer/RNG checkpoint,
   source-aligned development payload, observed trace, role history and
   independently retained ten-epoch recovery chunks. A failed/partial job is
   infrastructure evidence, never a fabricated completed comparison.

The post-run acceptance must verify source/cache/sidecar/runtime/transform,
all forty trace rows, finite predictions and loss, best-checkpoint hash,
50,000 exact development source indices and target hash, protected-role flags,
and actual cost. A strict/replay-ready RML terminal requires a genuinely
validated prospective prelaunch, aligned reference payload and all native
evidence; do not infer missing fields. Training completion alone is not RML
closure.

## Decision and boundaries

For this prospective question, nomination requires a positive aligned paired
interval and at least 0.003 eV development gain versus K1; this 0.003 is a
**study-specific materiality choice**, not an immutable V5-wide threshold or
measured run variance. The extra parameters and elapsed GPU time must be
reported. A favorable 100K result is only a shortlist signal: any disjoint
fixed500K NO_TRAIN audit, second seed, 500K training or desktop handoff needs a
separate explicit compute/role decision. A negative or nonportable result
closes this exact adaptation; it does not falsify the published model family.

Alternatives to record: 2/3-hop path counts may duplicate information in K1's
nine local layers; virtual-state update may be a weaker global bandwidth than
K1's learned slot; 5.27M parameters may overfit 100K; and a benefit on the
reused development role may not transfer. This protocol cannot justify a
later hyperparameter/seed sweep without a new hypothesis and budget decision.

## Bounded execution retry after preflight diagnosis

The v1 physical attempt stopped before epoch 0 because its first-step runtime
projection was not representative. A separately accepted train-role-only
[runtime profile](results/profile_v2_decision.md) measured the same 4 x 256
model under FP32/BS128 on the actual assigned T4 and passed the six-hour
wall/memory gate. The v2 attempt uses a new run/trajectory/source identity and
the warmed-step estimator; its dataset, seed, model, optimizer, schedule,
sample exposure and protected-role boundary do not change. The profile has no
development result and cannot itself support a scientific claim. The v1
prospective record stays frozen as an infrastructure-ended attempt.
