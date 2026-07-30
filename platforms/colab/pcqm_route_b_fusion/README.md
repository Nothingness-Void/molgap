# PCQM Route B Fusion on Colab

Upload the complete local payload directory to:

`MyDrive/MolGap/results/pcqm_route_b_fusion_payload_20260729/`

It contains the immutable payload, four checkpoints, runtime wheel, and runner.
Upload `molgap_pcqm_route_b_fusion.ipynb` to
`MyDrive/MolGap/notebooks/`, then open it in Colab.

The notebook writes atomic seed/epoch checkpoints below
`MyDrive/MolGap/checkpoints/pcqm_route_b_fusion_tuned_gps9_20260729/` and
small accepted summaries below
`MyDrive/MolGap/results/pcqm_route_b_fusion_tuned_gps9_20260729/`.

The run is development-only. It does not read official-validation labels,
official test, or the sealed 20K.
