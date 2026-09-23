# Execution status

Kaggle3 kernel `nvoid912/molgap-gptrans-pair-norm-v5-s42` version 2 completed
from source commit `5f8d27e52d73dca782fad19cde63896e590e8245` on
2026-09-18. Both arms completed 60 epochs and passed frozen no-inference
acceptance. `pair_update_norm` passed the shortlist gate; `pair_post_norm` did
not. The scientific disposition is in `decision.md` and exact values are in
`acceptance.json`.

Version 1 confirmed two Tesla T4 devices, then failed before training because
the accepted graph cache referenced the missing `molgap.pcqm_wedge.WedgeData`
pickle class. Version 2 restores that compatibility module without changing
the graph cache, frozen initialization, model variants, or training contract.

The raw output remains under
`platforms/_records/kaggle/training/pcqm_gptrans_pair_norm_v5_s42_v2/`.
