# Unclassified SchNet Download, 2026-06-12

These files were recovered from `D:\下载` on 2026-07-22. They form a compatible
SchNet architecture (`hidden_channels=192`, `num_interactions=6`) but no reliable
run identity was present, so they are deliberately not registered.

| File | Observed metadata | SHA-256 |
|---|---|---|
| `schnet_training_checkpoint_epoch75.pt` | Epoch 75 training checkpoint; best epoch 50; best validation MAE 0.120275 eV; includes optimizer/scheduler/scaler state | `19DDA3033AB448D2A57AC40DD22254712A703910239ED09BCAC7FEF07CD27475` |
| `schnet_best_model.pt` | Bare 92-tensor model state with the same architecture; relationship to the training checkpoint is unproven | `30297032E6DDF12F12625B56EF93010BA2DE13E20661E80BE23499B5752A6B56` |

Do not promote either file without reconstructing its dataset, target scaling,
split, and evaluation record.
