# MolGap Notes

## Path update
- Production pipeline scripts now live under `scripts/pipeline/`.
- Evaluation and validation scripts now live under `scripts/evaluation/`.
- Experimental benchmark scripts now live under `scripts/experiments/`.
- Deferred application scripts now live under `scripts/todo/`.
- Shared utilities moved from `src/utils.py` to `src/molgap/utils.py`.
- Historical archive notes may still mention the pre-refactor `src/*.py` paths.

## Conversation notes

### Raw data count question
The user noticed the raw CSV has only about 280 rows and asked whether this is before or after filtering.

Answer:

- It is after filtering by the current script.
- But it is not the final full filtered PubChemQC subset.
- It was produced from partial Range reads, currently only the front chunk of each JSON file.
- Therefore it should be treated as sample/smoke-test data.

### Estimated data needs
Recommended data sizes:

```text
~300 rows: pipeline only
1,000 rows: preliminary model sanity check
5,000–10,000 rows: first usable baseline
30,000–100,000 rows: stronger report-quality modeling
50,000+ rows: better for scaffold split / embedding / GNN experiments
```

First formal target: 10k+ filtered rows.

### Embedding discussion
The user asked about using embeddings from:

```text
ChemBERTa
MolBERT
Mol2Vec
GROVER
Uni-Mol
MolFormer
```

Recommendation:

1. Do not start with embeddings.
2. First build Morgan fingerprint + RDKit descriptor baseline.
3. Then add ChemBERTa and MolFormer as the first embedding comparisons.
4. Then try fusion: Morgan + RDKit + embedding.
5. Later consider Mol2Vec / MolBERT.
6. GROVER / Uni-Mol are advanced and more complex, especially Uni-Mol because it is more naturally 3D/coordinate-based.

### Recommended embedding experiment design

```text
Traditional baseline:
Morgan2048 + RDKit descriptors -> ExtraTrees/LightGBM

Embedding baseline:
ChemBERTa -> Ridge/ExtraTrees/LightGBM
MolFormer -> Ridge/ExtraTrees/LightGBM

Fusion:
Morgan2048 + RDKit + ChemBERTa -> LightGBM/ExtraTrees
Morgan2048 + RDKit + MolFormer -> LightGBM/ExtraTrees
```

Use the same fixed split across all experiments.

### Pooling recommendation
For transformer embeddings, use attention-mask mean pooling rather than naïvely averaging padded tokens.

## Coding style preferences inferred from user/project
- User prefers incremental development from simple to complex.
- User wants persistent local project notes so context survives session closure.
- User has prior molecular ML experience and can understand RDKit descriptors, embeddings, split strategy, and model comparison.
- Keep project scientifically defensible: fixed splits, clear metrics, and avoid overclaiming small-sample results.

## Next assistant should do
If the user says to continue implementation, start with:

```text
src/molgap/utils.py
scripts/pipeline/02_clean.py
```

Then verify with:

```bash
python scripts/pipeline/02_clean.py
```

After that implement:

```text
scripts/pipeline/03_features.py
scripts/pipeline/04_train_baseline.py
```

Do not jump directly to embeddings or GNN until traditional baseline is complete.

