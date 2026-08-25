# Stage 6 Embedding Experiments Archive

Archived at: 2026-06-03.

## Implemented files

```text
scripts/colab/molgap_embeddings.ipynb    (Colab notebook, T4 GPU)
scripts/colab/colab_extract_embeddings.py (standalone .py version)
scripts/experiments/14_train_with_embeddings.py (local comparison)
```

## Embedding models used

- ChemBERTa: seyonec/ChemBERTa-zinc-base-v1 (768-dim)
- MolFormer: ibm/MoLFormer-XL-both-10pct (768-dim)

## Generated data

```text
data/processed/embeddings_chemberta.csv  (10000 x 769)
data/processed/embeddings_molformer.csv  (10000 x 769)
data/processed/embeddings_all.csv        (10000 x 1537)
```

## Results ranking (by avg MAE, random split test)

```text
1. traditional_lightgbm          MAE=0.1587  R2=0.9099
2. fusion_chemberta_lightgbm     MAE=0.1681  R2=0.8979
3. fusion_molformer_lightgbm     MAE=0.1715  R2=0.9001
4. fusion_both_emb_lightgbm      MAE=0.1747  R2=0.8950
5. traditional_ridge             MAE=0.2170  R2=0.8542
6. both_emb_only_lightgbm        MAE=0.2356  R2=0.8311
7. molformer_only_lightgbm       MAE=0.2450  R2=0.8183
8. chemberta_only_lightgbm       MAE=0.2868  R2=0.7545
```

## Key findings

1. Traditional features (Morgan fingerprint + RDKit descriptors) are the best standalone feature set.
2. Embedding-only models are much worse (MAE 50%+ higher than traditional).
3. Fusion (traditional + embedding) does not improve over traditional alone — the extra dimensions add noise.
4. MolFormer embeddings are slightly better than ChemBERTa for this task.
5. Ridge is more sensitive to feature quality; LightGBM handles noisy features better but still can't overcome the gap.

## Interpretation

For CHON small molecules (MW 200-300) with 10k samples, hand-crafted molecular descriptors capture the relevant structure-property relationships more effectively than pretrained language model embeddings. This is likely because:
- The molecular space is narrow (only CHON, small MW range)
- RDKit descriptors directly encode chemically relevant properties
- Pretrained embeddings are trained on broader chemical space and don't specialize

## Final model decision

Tuned LightGBM + Morgan fingerprint + RDKit descriptors is the production model. Embeddings are not used.
