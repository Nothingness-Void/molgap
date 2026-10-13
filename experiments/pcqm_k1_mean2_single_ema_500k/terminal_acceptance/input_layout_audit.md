# Input-layout evidence boundary - 2026-10-13 JST

The authoritative Kaggle1 dataset file listing has13 entries, including
root `manifest.json` and11 `train/train_shard_*.pt` assets. No graph or label
bytes were downloaded in this audit. See [file inventory](graph_dataset_file_inventory.json).
This establishes current publication shape; it does not establish the failed
worker's actual mount directory or the mounted version.

The retained recipe fixes the relative path to
`pcqm4mv2-ogb-fixed-500k-scnet-v1/manifest.json`. The generic bootstrap passes
`/kaggle/input` without resolving that dataset's actual mount. The loader
rejects the missing joined path before hash verification or deserialization.
Local release checks cover source inputs, imports and initial state; the
retained release does not certify this remote graph filesystem path.

Existing legacy owners already locate manifests by identity, for example
`gptrans_screen_workflow._owner_kwargs` uses a unique SHA-matching metadata
manifest under the input root before its graph owner validates bytes. Its
family-specific runtime wrapper is not a drop-in shared trainer. Any separately
authorized correction should reuse or extend the existing input-resolution
owner with unambiguous hash checking and flat/nested mount coverage, preserving
the scientific recipe. No such correction or submission was made during this
acceptance.
