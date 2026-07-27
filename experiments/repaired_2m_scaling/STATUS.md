# Repaired-2M scaling — live remote status

Dated operational state for this experiment's remote jobs. Conclusions live in
the decision records under `results/`; the recommended production model lives in
`CURRENT_STATE.md`. Nothing here is a promotion.

## Kunshan 3D graph construction

**Primary view: accepted.** Job `117854186` stopped after 36/100 atomic shards
because the original-1M cache was loaded before `fork`, causing copy-on-write
memory amplification. The isolated-`spawn` repair job `117872652` resumed the
same immutable directory and finished all 100 shards in `07:32:46` at `23.47 GB`
peak RSS, reconciling 2,000,000 requested rows as 869,142 reused, 1,119,974
newly built, and 10,884 failed conformers — 1,989,116 graphs.

Acceptance-only job `117958648` then strictly accepted all 100 shards in
`00:30:21`: 1,989,116 unique graphs, 10,884 recorded conformer failures,
complete source-target alignment, and matching graph/sidecar hashes. The
retrieved manifest SHA256 matches the remote copy. Process completion alone was
never treated as accepted evidence.

Two validator-only defects were fixed on the way there. The first acceptance
code wrongly required graph-local `cid` and SMILES; repaired-2M graphs
intentionally store `source_idx`, coordinates, and labels, with identity in the
immutable source CSV, so acceptance now resolves CID/SMILES by `source_idx` and
checks all three labels against the CSV. The second was a sidecar-location
mismatch: the real builder writes all 100 sidecars under
`graph_shards/reports/`, not beside the graph files.

**Secondary view: running.** Seed-`314159` build `117966453` is on compute node
`j05r4n04` with atomic 20K-row shards and resume enabled. **Do not duplicate
it.** The two earlier failed attempts (`117857094`, `117950883`) produced no
secondary shards and did not modify the primary cache.

Records: `platforms/_records/scnet/kunshan_repaired_2m_3d/`.

## Colab fallback

The Colab primary build was stopped after 25/100 durable shards (500,000 source
rows) because the capped eight-worker CPU path was too slow; no complete shard
was lost. The resumable notebook and wheel bundle stay under
`platforms/colab/repaired_2m_3d/`. An identity audit found 871,693 repaired rows
already in the original 1M corpus, so the builder remaps reusable coordinates by
CID plus canonical SMILES and constructs only missing identities.

Full repaired-2M SchNet training remains **disabled** in the notebook. The fixed
30K pilot showed ordinary-FusionHead regression, so training needs the explicit
bounded-residual gate token after the PubChemQC 100K two-SchNet screen is
accepted. Plan: `results/3d_colab_plan.json`.

## Frozen 3D protocol

The full-scale Route B primary SchNet is frozen at `176/160/6`, cutoff `10 A`,
dropout `0.05`. Its split reproduces the GPS seed-42 roles across all 2,000,000
source rows **before** filtering failed ETKDG molecules; the previously used
filtered-list split is forbidden for fusion because it misaligns encoder roles.

## Blocked next stage

The hierarchical 2D+3D contract is implemented but not submitted. Either the
fixed GPS7+GPS9 equal blend or the frozen three-GPS dense prediction is the exact
identity path, and primary plus augmented lightweight SchNet embeddings may only
add a `+-0.10 eV` bounded correction. It uses a new scaffold-disjoint split
inside untouched base-model test rows and stops if fewer than 95% of those rows
carry both 3D views. It stays blocked on accepted primary/secondary caches, two
trained SchNet checkpoints, and their embedding parts.
