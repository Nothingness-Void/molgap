# Geometry-channel screen status

The source dataset was published privately on Kaggle1 and reached `ready`.
The Kaggle accelerator adapter submitted the paired T4x2 kernel. Kaggle
assigned the actual kernel reference
`nothingnessvoid/molgap-geometry-channels-100k-s42-v1` (kernel ID 136150940),
which differs from the requested metadata slug. The authoritative status
query returned `RUNNING`. The pulled entry script matched the frozen local
script after line-ending normalization, and its two input dataset references
matched the release gate. The actual reference is bound in
`launch/platform_response.json` and the shared CLI launch receipt.

No remote GPU preflight, training completion, MAE, or replay acceptance is
asserted yet. Both arm outputs must pass the protocol's independent terminal
acceptance before any scientific decision or branch routing.
