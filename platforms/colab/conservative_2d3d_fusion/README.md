# Conservative 2D + 3D Fusion on Colab

This package runs the Track C P1 screen over frozen repaired-2M 2D predictions
and accepted dual-SchNet embeddings. It never runs either encoder on Colab.

## Drive layout

Upload the notebook and wheel to:

`MyDrive/MolGap/notebooks/molgap_conservative_2d3d_fusion_r1/`

Upload the four immutable inputs to:

`MyDrive/MolGap/results/molgap_conservative_2d3d_fusion_r1/inputs/`

- `training_payload.pt`
- `training_manifest.json`
- `external_payload.pt`
- `external_manifest.json`

The notebook writes epoch checkpoints under
`MyDrive/MolGap/checkpoints/molgap_conservative_2d3d_fusion_r1/` and accepted
metrics, predictions, and manifests under
`MyDrive/MolGap/results/molgap_conservative_2d3d_fusion_r1/run/`.

## Resume contract

Run all cells in order on an A100. The first cell mounts Drive before doing any
work. Every epoch updates an atomic `.last.pt` checkpoint. Completed seeds are
reused only when their input hash, configuration, and model hash match. Running
the training cell again resumes an interrupted seed and skips accepted seeds.

Six small heads are evaluated: equal GPS7/GPS9 and dense GPS7/GPS9/GPS11 bases,
each with seeds 42, 43, and 44. Pure 2D is the exact validation fallback. The
fixed external gate contains 998 OOD and 975 P8-hard molecules; sealed 20K is
not present.
