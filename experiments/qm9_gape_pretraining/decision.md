# GAPE-lite QM9-30K decision — 2026-09-09

## Evidence

Kaggle2 kernel `kaseichou/molgap-qm9-gape-lite-s42`, version 2, completed on
the accepted 30,000/3,000 QM9 cache. Independent no-inference acceptance
recomputed every validation payload MAE and every artifact hash. The executable
comparability guard confirmed that all three arms shared one task, Kaggle2,
Tesla T4 hardware, split `62f1cdefdaec6877`, seed 42, FP32, physical batch 128,
AdamW settings, cosine schedule, and 40-epoch labeled exposure. No QM9 test or
PCQM role was read.

| Arm | Validation Gap MAE (eV) | Inference parameters | Best epoch |
|---|---:|---:|---:|
| Fresh EdgeState GPS9 + RWSE16 | 0.130141363 | 4,771,073 | 35 |
| Shuffled-alignment PE control | 0.129175797 | 4,777,281 | 38 |
| Matched GAPE-lite PE | 0.130241096 | 4,777,281 | 38 |

The matched PE regressed by `0.000099733 eV` against the fresh baseline and by
`0.001065299 eV` against its equal-compute shuffled-alignment control. It did
not approach the frozen `0.003 eV` promotion gate. The shuffled control's
`0.000965565 eV` apparent gain over baseline is below the gate and below the
previously observed scale of single-run drift, so it is not a positive result.

Version 1 is excluded from scientific interpretation: it stopped before the
first epoch because PyTorch 2.6 changed the default trusted-object loading
mode. Version 2 changed only that compatibility setting.

## Attribution

The accepted EdgeState GPS9 already receives RWSE16 and repeatedly propagates
real-bond states. A small constant-input GAT trained to recover node alignment
therefore mostly learns local degree and neighborhood identities that are
redundant with the existing topology channels. The alignment objective rewards
node distinguishability and perturbation stability, not electronic relevance.
The equal-compute control outperforming the matched objective is direct
evidence that the correspondence signal itself added no useful Gap information
in this contract.

The result rejects this exact three-layer, 32-dimensional, edge-drop-only
GAPE-lite implementation. It does not claim to reject the paper's full
Hungarian/BCE method, a transductive all-topology setup, or another downstream
model that lacks RWSE and bond features. Those are materially different
questions and cannot rescue this result by retuning its seed, width, noise, or
pretraining length.

## Decision

Close this GAPE-lite question. Do not submit PCQM-100K, another seed, a larger
generator, another noise rate, shadow evaluation, full training, official
evaluation, or IMS work from this result. Return Track C to selecting one
materially different information-flow mechanism under the same global
batch-128 comparability policy.

Compact machine evidence is `results/seed42_metrics.json`; complete models,
checkpoints, payloads, traces, logs, and acceptance remain under
`platforms/_records/kaggle/training/qm9_gape_lite_s42_v2/`.

