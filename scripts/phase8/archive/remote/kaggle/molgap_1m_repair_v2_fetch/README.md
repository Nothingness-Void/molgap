# 1M-v2 Candidate Fetch

This CPU Kaggle job fetches a disjoint 600K PubChemQC B3LYP candidate pool for
the Phase 8 1M-v2 repair as ten independently retrievable 60K rounds. It
excludes every CID and canonical SMILES in the rejected 1M continuation
mounted from `nothingnessvoid/1m-full`, then runs four independent chemistry
groups concurrently within a bounded round.

The job has Internet enabled because PubChemQC source records are read from the
Hugging Face dataset via HTTP range requests. It deliberately requests no GPU:
RDKit descriptors and candidate filtering are CPU-bound.

## Checkpoint Contract

One completed round emits four CSVs, final reports, per-group progress JSON,
logs, a manifest with SHA-256 checksums, and `phase8_repair_v2_round_XX.zip`.
The collector flushes its CSV and atomically updates progress every 250 rows.
Kaggle does not publish files from a cancelled task, so only a normally
completed round is a valid restart point.

After downloading and validating a successful round, publish its four CSVs and
manifest into a private Kaggle checkpoint dataset. Package the next round with
that dataset mounted so its CIDs and canonical SMILES are excluded:

```powershell
.venv\Scripts\python.exe scripts\phase8\archive\remote\kaggle\molgap_1m_repair_v2_fetch\package_kernel.py `
  --out-dir $env:TEMP\molgap-repair-r02 --round-index 2 `
  --checkpoint-dataset nothingnessvoid/molgap-phase8-repair-v2-checkpoints
kaggle kernels push -p $env:TEMP\molgap-repair-r02 -t 43200
```

Use `stage_checkpoint_dataset.py` before `kaggle datasets create` (round 1) or
`kaggle datasets version` (later rounds). It refuses to stage a round whose
hashes, quotas, or disjointness checks fail and writes the required Kaggle
dataset metadata. For round 2 and later, pass every already accepted checkpoint
CSV as `--exclude-csv`; the staged round is rejected if any CID or canonical
SMILES overlaps. For example:

```powershell
.venv\Scripts\python.exe scripts\phase8\archive\remote\kaggle\molgap_1m_repair_v2_fetch\stage_checkpoint_dataset.py `
  --round-dir D:\downloads\phase8_repair_v2_round_01 `
  --train-csv data\raw\phase8_expansion_1m.csv `
  --checkpoint-dir data\raw\phase8_repair_v2_checkpoint_dataset `
  --dataset-id nothingnessvoid/molgap-phase8-repair-v2-checkpoints
kaggle datasets create -p data\raw\phase8_repair_v2_checkpoint_dataset
```

Before staging a parallel round, run `reconcile_round.py`. It records and
removes any same-round cross-bucket duplicates deterministically, rather than
silently accepting duplicate training rows.

Never submit multiple rounds simultaneously: each round must consume the
verified checkpoint dataset created by the previous one.
