# RTX 5060 speed-only check

User scope: fixed 100K TRAIN, ten epochs per arm, no development or MAE,
no RML publication. This is engineering timing, not scientific qualification.

Reuse `molgap.k1_local_speed` epoch mode and the existing K1 model, train step,
epoch loader, source bundle and atomic IO. Reference uses native AdamW,
whole-batch transfer and per-batch scalar accumulation. Candidate uses fused
AdamW, CPU layout, selective transfer and epoch-end scalar accumulation.
Both retain FP32/TF32-off, BS128, the same initialization/order, and the original
40-epoch cosine schedule stopped at ten epochs. Alternate epoch order on one GPU.
Report training-pipeline time separately from source/data preparation and saves.

`prepare.py` freezes inputs/source; run the frozen `molgap.k1_local_speed` CLI.
Local raw output is under ignored `platforms/_records/local/staging/`.
One nonexclusive desktop run cannot establish quality or T4/A100 speed.

The earlier local quality-prefix preparation was canceled before any training
or prospective publication. Its unpublished planning drafts were removed.
