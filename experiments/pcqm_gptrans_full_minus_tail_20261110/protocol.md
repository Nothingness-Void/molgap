# GPT-T full-train without the tail holdout

2026-11-10 JST. Desktop-owned Track B. One frozen GPT-T BondLocal+PairNorm recipe
(gptrans_t / degree_bond_local_pair_update_norm_ema999, lr 1e-3, wd 0.05,
warmup 4, cosine 60 passes, EMA 0.999, final-ema, batch 128, seed 42) trained
from the retained fresh initialization on TRAIN rows [0, 3,250,000) only.

## Question

Does a 60-pass full-scale GPT-T endpoint come out usable for a later fixed
fusion, and, from snapshots scored offline on tail_holdout_v1, where does
generalization stop improving? The earlier package trained on every row, so its
tail could not be used as evidence.

## Data

Derived manifest of the accepted full view (parent SHA256
7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d): roles.train is
[0, 3,250,000) over shards 0000-0064 (each 50,000 rows, so the cut is aligned);
no development role. tail_holdout_v1 = rows [3,250,000, 3,378,606) is excluded and
only scored offline. Normalization constants are recomputed from the 3,250,000 train
labels by the existing loader. Optimizer steps 25,390 per pass x 60 = 1,523,400
(194,995,200 presentations, drop_last).

## Accelerator

The frozen spec declares "A100/L4"; the trainer accepts either. Wall times from
different GPU types are not comparable.

## Diagnosis

The notebook copies the rolling last_checkpoint.pt to a snapshots directory. The
offline evaluator scores each snapshot on tail_holdout_v1. Curves are read, never
used to pick a snapshot. Official validation is for the finalist only, with explicit
per-use authority.
