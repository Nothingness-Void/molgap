# Isolated G1 decay-coefficient submission

On 2026-10-04 the shared preparation command reused the accepted G1 EMA999
reference, frozen initialization, immutable fixed100K cache and existing GPTrans
runner. Local preparation took 47.58 seconds; design, tests, publication and
monitor handoff are outside that measurement. Mechanical tests passed 35 cases
without local model construction, training or inference.

The local package `platforms/_records/kaggle/packages/gptrans_g1_decay_clock_v1/release`
passed source/recipe/initialization/upload checks. Kaggle3 returned the exact
[v1 physical receipt](submission_v1.json), without invalid mounts or submission
uncertainty. [Remote entry verification](remote_kernel_verification.json) records
SDK-local line-ending normalization and the observed RUNNING status. This status
does not prove remote optimizer calibration or epoch completion.

Only the all-parameter AdamW coefficient changes under the [protocol](../protocol.md).
The baseline is not retrained. Immutable reference evidence is in
[`reference`](../reference/reference_bundle.json); the prospective optimizer-only
[prelaunch assessment](degree_decay001_ema999/comparison_readiness_prelaunch.json)
passed actual repository evidence binding, not just an ID/digest check.

[Source publication](source_publication_v1.json), [prospective trajectory](degree_decay001_ema999/rml_plan/trajectory.json)
and [monitor binding](monitor_binding.json) preserve the bounded chain. The wrapper
`../accept.py` reuses saved-tensor verification and per-arm RML closure. Actual
STRICT_CAUSAL and Replay-Ready status require post-run evidence and pool admission;
neither was claimed at launch. No coefficient grid, extra seed, scale or full
training successor was authorized.
