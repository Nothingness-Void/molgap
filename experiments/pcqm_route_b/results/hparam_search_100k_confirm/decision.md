# PCQM Route B 100K Hyperparameter Confirmation

The four encoder searches completed on IMS on 2026-07-29. Each reported value
is the mean of scaffold-development Gap MAE over seeds 42, 43, and 44. The
official validation metric, official test set, and sealed 20K were not used.

| Encoder | Winner | Mean Gap MAE (eV) | Seed std (eV) |
|---|---|---:|---:|
| GPS9-192 | trial_02 | 0.204502 | 0.000583 |
| GPS11-160 | trial_09 | 0.200543 | 0.000822 |
| Primary SchNet | trial_11 | 0.172015 | 0.001612 |
| Augmented SchNet | trial_09 | 0.162460 | 0.000129 |

## Frozen winners

| Encoder | Learning rate | Weight decay | Dropout | Batch | Warmup | Grad clip |
|---|---:|---:|---:|---:|---:|---:|
| GPS9-192 | 1.6e-4 | 1e-5 | 0.05 | 256 | 0.05 | 1.0 |
| GPS11-160 | 1.6e-4 | 1e-6 | 0.00 | 384 | 0.10 | 0.5 |
| Primary SchNet | 6e-4 | 3e-6 | 0.05 | 64 | 0.10 | 2.0 |
| Augmented SchNet | 8e-4 | 1e-6 | 0.00 | 192 | 0.10 | 0.5 |

The augmented SchNet was the strongest and most stable 100K encoder. Its mean
development MAE was 0.009555 eV lower than the primary SchNet and its seed
standard deviation was the smallest of all four encoders. GPS11-160 was
0.003959 eV better than GPS9-192.

These values select full-scale training configurations only. They are not
official PCQM validation results and are not comparable to accepted
common/OOD/P8-hard production metrics. Fusion remains blocked until the
full-scale checkpoints and aligned embeddings are independently accepted.
