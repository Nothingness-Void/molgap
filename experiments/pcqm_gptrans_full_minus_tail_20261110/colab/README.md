# GPT-T minus-tail Colab package (2026-11-10)

Builds `MolGap_GPTrans_MinusTail.ipynb` with `python build_notebook.py` (never run locally).
One frozen GPT-T arm (B128, 60 passes, 1,523,400 steps) on TRAIN rows [0, 3,250,000); the tail
`tail_holdout_v1` (rows 3,250,000..3,378,605) is excluded and scored offline. The spec declares
`A100/L4`, so either GPU passes the trainer check; times from different GPUs are not comparable.

## Drive layout (MyDrive/MolGap/V5/)

| Local file | Drive path |
|---|---|
| `D:/下载/MolGap-Colab-MinusTail-20261110/gptrans_pairnorm.zip` (22,609,255 B, sha b7ee97ae...) | `releases/gptrans-minus-tail-20261110/gptrans_pairnorm.zip` |
| `D:/下载/MolGap-Colab-MinusTail-20261110/manifest_minus_tail_v1.json` (sha 100716d0...) | `fixed_data/full_train_v1/manifest_minus_tail_v1.json` |
| `D:/molgap-artifacts/colab-full-20261108/data_archives/data_archives.json` and `train_topology_000..022.zip` | `fixed_data/full_train_v1/archives/` (unchanged from the earlier package; only shards 0000-0064 are unpacked) |

Snapshots: every 3 completed passes the notebook copies the rolling `last_checkpoint.pt` to
`runs/gptrans-minus-tail-20261110/gptrans_pairnorm-snapshots/pass_NNN.pt` (about 94 MB each, about 20 files).
Read epoch 1 wall time from the log; projected cost is that x 60.
