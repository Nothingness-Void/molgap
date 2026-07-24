# MolGap Additive 2M Candidate Fetch

This CPU-only Kaggle job prepares fresh candidates for a future additive
1.5M-to-2M experiment. It preserves the complete 1.5M training table and
excludes both that table and the complete 726,966-row prior repair candidate
union by CID and canonical SMILES.

Six durable 60K rounds target a fresh 360K pool. Each round is 80% balanced
general chemistry and 20% capped targeted buckets. The fresh pool is later
combined with the approximately 227K candidates not selected for the 1.5M
top-up, then exactly 500K rows are selected under a separate audit.

Sequential rounds must mount the validated output dataset from every accepted
prior round and exclude all of their CSVs. Two rounds may run concurrently only
when `SOURCE_SHARD_COUNT` is identical and their stable `SOURCE_SHARD_INDEX`
values differ. This partitions the sorted Hugging Face source-file list before
random window selection, so concurrent tasks cannot scan the same source file.

Every task writes atomic per-group progress, a final manifest with the source
shard identity, and one independently downloadable ZIP. A later round must
mount and exclude both concurrent outputs.

`fetch_general_overnight.py` is the separate untargeted fallback. It accepts any
in-domain molecule that passes the global element, molecular-weight, positive-
Gap, CID, and canonical-SMILES checks. Its 500K target is split into five 100K
chunks with one ZIP per chunk. It may overlap concurrent rounds that were not
yet available as checkpoint datasets, so cross-run deduplication is mandatory
before any row count is accepted.

Every continuation must use a new `--run-tag` and `--seed-base` when packaged.
This prevents output-name collisions and moves the random source windows rather
than replaying the previous overnight scan.

The `hard` packaging profile uses `sampling_spec_hard.json`, 24 windows per
source file, and no balanced-general rows. Its four groups target the fixed
residual-analysis regions: very-large high-sp3/non-aromatic/flexible molecules,
very-large macrocycles/multi-amides, flexible low/mid-Gap molecules, and
high-sp3 non-aromatic molecules. Partial completion is expected if the remaining
source is exhausted.

The `complementary` profile uses `sampling_spec_complementary.json`. It excludes
all accepted broad/general/hard checkpoints and targets high-Gap rigid/hetero,
small hetero-dense, sulfur/halogen-rich, bridged/fused rigid, and conjugated
donor-acceptor regions that were not explicit objectives of rounds 01--05.
Run it as two stable source shards and reconcile them before accepting counts.
