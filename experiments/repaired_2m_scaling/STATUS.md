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

**Secondary view: accepted.** Seed-`314159` build `117966453` finished all 100
atomic 20K-row shards on Kunshan in `11:16:59`; acceptance job `118050455` then
accepted them in `00:44:56` — 1,986,868 unique graphs from 1,989,116 requested
(2,248 conformer failures), `source_idx` and CID unique, every label matching the
source CSV.

That acceptance needed its own contract. Secondary shards are keyed to the
primary view's per-shard graph counts rather than to source-index spans, so their
sidecars carry no `start`/`stop` and the primary function raises `KeyError`. Two
properties matter only here: each shard names the `primary_shard_sha256` it
derives from, proving both views cover the same molecule set; and **all 1,986,868
paired coordinates differ from the primary view** (distinct fraction `1.0`), which
is the one check a copied cache could not pass. Contract:
`src/molgap/artifact_acceptance.py`.

Two earlier Kunshan attempts (`117857094`, `117950883`) produced no secondary
shards and did not modify the primary cache. The IMS CPU array was a parallel
route: preflight `1091036` accepted framework-neutral shard 99, but array
`1091138[]` briefly reached 23 concurrent 16-CPU tasks and was stopped to respect
the shared-cluster boundary. Twenty-four raw shard reports survived and are no
longer needed now that Kunshan completed the view; any future IMS array stays
capped at four concurrent tasks.

Records: `platforms/_records/scnet/kunshan_repaired_2m_3d/` and
`platforms/_records/ims/repaired_2m_secondary/`.

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

The Track A full-scale primary SchNet is frozen at `176/160/6`, cutoff `10 A`,
dropout `0.05`. Its split reproduces the GPS seed-42 roles across all 2,000,000
source rows **before** filtering failed ETKDG molecules; the previously used
filtered-list split is forbidden for fusion because it misaligns encoder roles.

## Next stage

Both 3D views are now accepted, so the graph-cache half of the hierarchical
2D+3D gate is satisfied. 1,986,868 of 2,000,000 source rows (99.34%) carry both
views, well past the contract's 95% floor.

The hierarchical contract itself is implemented but not submitted. Either the
fixed GPS7+GPS9 equal blend or the frozen three-GPS dense prediction is the exact
identity path, and primary plus augmented lightweight SchNet embeddings may only
add a `+-0.10 eV` bounded correction, on a new scaffold-disjoint split inside
untouched base-model test rows.

The bounded-residual execution gate is now open for the final freeze sprint.
Shard-streamed SchNet jobs were submitted on IMS: primary and augmented
preflights `1102071/1102072`, followed by full training jobs `1102073/1102074`
with scheduler `afterok` dependencies. Each full job writes an atomic
epoch-level checkpoint and refuses a resume whose input/model contract changed.

Remaining blockers: independently accept both checkpoints, export and accept
their source-aligned embedding parts, then run the bounded residual fusion gate.
The export jobs are already dependency-chained as `1102137/1102139`; CPU
acceptance job `1102141` starts only if both exports succeed.

Both accepted views were subsequently staged to the IMS root
`/lustre/home/users/sm2/chou/molgap-repaired-2m-3d`. A full transfer audit
rehashed all 100 graph shards and 100 sidecars in each view against the Kunshan
acceptance manifests. One secondary shard affected by a duplicate local transfer
process was recopied and the entire secondary view was rehashed; the final audit
has zero mismatches. Evidence:
`platforms/_records/ims/repaired_2m_graph_transfer/acceptance.json`.
Submission evidence:
`platforms/_records/ims/repaired_2m_graph_transfer/submission.json`.
