# Naming Convention

This file is the single source of truth for repository naming. Names should let
a human identify an artifact's role and scope without opening it.

## General rules

- Use lowercase ASCII `snake_case` for directories and files.
- Name a directory by its role, model composition, or experimental question.
- Do not use dates or calendar phases in active directory names.
- Do not use ambiguous words such as `new`, `latest`, `final`, `best`, `tmp`,
  `test2`, or `misc`.
- Put dates inside decision records and manifests, not in directory names.
- Keep historical names unchanged only when they are frozen provenance. New
  pointers must use the current path.

## Model result directories

Use:

```text
[routed_]encoder[_encoder...]_training_rows_version
```

Examples:

```text
gps7_schnet_500k_v3
routed_gps7_gps9_schnet_500k_v4
gps11_160_gps9_schnet_100k_v1
```

Rules:

- List encoders in inference/data-flow order.
- Add `routed_` when not every encoder runs for every molecule.
- Use architecture tokens such as `gps7`, `gps9`, `gps11_160`, `gine6`,
  `schnet`, and `tensornet`.
- `30k`, `100k`, `500k`, `1m`, and `2m` mean training molecules, never model
  parameters.
- Parameter counts belong in `summary.json` or `metrics.json`, not in names.
- `vN` is the accepted model generation. It is not a seed, retry number,
  checkpoint epoch, or experiment attempt.
- Bump the version only when an accepted change alters training data,
  architecture, routing behavior, targets, or the shipped inference contract.

Retired production evidence belongs under:

```text
production/03_train/_retired/<model_result_directory>/
```

## Experiment directories

Name one directory after one question:

```text
qm9_architecture
conformer_protocol
schnet_arch
pubchemqc100k_architecture
```

Do not encode a conclusion in the directory name. Record the conclusion in
`decision.md`.

Use this run identity when multiple controlled runs share one question:

```text
n<train>_<validation>_<test>/<candidate>_<geometry>/seed<seed>/
```

Example:

```text
n100000_10000_10000/schnet_angle_dihedral_etkdg/seed42/
```

## Standard file names

| File | Owns |
|---|---|
| `README.md` | Directory purpose and pointers |
| `decision.md` | Dated human-readable conclusion |
| `summary.json` | Compact machine-readable comparison |
| `metrics.json` | Metrics for one run |
| `manifest.json` | Inputs, hashes, alignment, and provenance |
| `checkpoint.pt` | Resumable training state |
| `model.pt` | Selected model weights |
| `predictions.csv` | Per-row predictions when retention is justified |
| `train.stdout.log` | Atomic run log |

Add a specific prefix only when several same-role files must coexist, for
example `common_metrics.json` and `ood_metrics.json`.

## Artifact placement

- Reusable code: `src/molgap/`.
- Thin command-line adapters: the owning tree's `scripts/`.
- Production decisions and compact metrics: `production/`.
- Question-specific evidence: `experiments/<question>/results/`.
- Large reproducible embeddings and graph caches: `data/cache/`.
- Selected checkpoints: `models/`.
- Platform submission and scheduler records: `platforms/`.
- Do not leave embeddings, predictions, and logs flat beside decisions unless
  they are intentionally retained and the directory `README.md` explains why.

## Compatibility

Legacy checkpoint filenames such as `phase8_*.pt` remain valid until a
checkpoint-manifest migration updates every registry entry and hash. New
directories and new artifacts must follow this convention.
