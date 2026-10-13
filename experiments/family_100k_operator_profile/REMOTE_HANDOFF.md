# Kaggle1 operator profile handoff - 2026-10-14 JST

Owner: `codex/exp/k1-gptrans-profiler-100k`, `D:/w/k1-gptrans-profiler`.
The user authorized fixed100K complete contracts plus profiling and shutdown
after verified submission. One physical Notebook, independent T4 per arm.

- Actual kernel: `nothingnessvoid/molgap-k1-gptrans-100k-profiler-s42`.
- Kernel ID138537758, version1; script/session version ID not returned.
- Authoritative scheduler RUNNING at2026-10-13T18:22:08.730371+00:00.
- Startup query returned empty log and no output files. Native allocation,
  qualification, training exposure and profile capture not yet observed.
- Private source: `nothingnessvoid/molgap-k1-gptrans-100k-profiler-s42-src`.
  Ready; all11 paths/sizes and download SHA256 values verified.
- Graph dataset: `nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1`.
  Accepted manifest identity remains `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.

Frozen executable source `faf5b6d13d94fa28a1d3e867bc17e453ca44da1e`.
Spec `cd2ec654371e0b08e539ec1e1022bc1e7381da15299ec195da348589c97b1252`.
Package `b1c931d1f87af5f601b2fb2c9faebaf17c729175937af2be1b73b9876df4be11`.
Source archive `0c6d43312114d08d7b44a3f5322db747a6aa86de568697192367f63e34ac1dbc`.
Remote API source text matches the submitted entry bytes; SDK `kernels_pull`
adds Windows CRLF when saving, so its file byte mismatch was a local newline
conversion. Raw API comparison and dataset membership both passed.

## Contracts and retained outputs

`k1_mean2`: existing K1/RWSE16 one-slot/nine-local-layer shape, mean2/live,
40passes31240updates. `gptrans_local`: retained degree-scaled256/pair32/12layer
true-bond64 model, EMA.999,60passes46860updates. See [selection review](evidence_review.md).
This is cost/operator diagnosis with contextual endpoints, not a strict
cross-family architecture comparison or model promotion.

[Config](config.json) binds7wall-hour ceiling including setup and180s cleanup.
Profile capture120warmup/16active spans checkpoint128. Profiling is also enabled
on one native two-step repeat; identical plain repeat/resume endpoints and actual
CUDA events are required by the all-arm barrier before formal training.
No automatic continuation, retry, monitor or server takeover.

Durable output root: `/kaggle/working/experiment/<arm>`.
Retrieve output manifest, best/last checkpoints, canonical trace, runtime/native
proof, aligned internal50K predictions and allocation accounting with existing
Kaggle/RML acceptance. For profiling also retrieve `execution_profile_receipt.json`,
its named `profiles/step_*/profile.json` and `trace.json`, plus
`profile_qualification/profile.json` and trace. Verify the context and all SHA
bindings; ordinary minimal family retention does not imply these extra files
were downloaded. Preserve the whole transaction directory if interrupted.
Official validation/test-dev/challenge remain forbidden.

On return reconcile exact version first, retrieve only relevant artifacts,
accept completeness/exposure and profile evidence, write attribution, then close
canonical RML through existing owners. Pending or STOP_FOR_COST is not a complete
endpoint; missing allocated/busy/queue time remains unknown. No new scientific
conclusion follows from the RUNNING snapshot.

## Local preparation notes

CPU source/registry regression207tests passed; after native profile wiring,
50targetedtests passed;7corruption bindings plus2release tests passed (9total).
Counts overlap; true CUDA profiling remains a remote qualification.
Local source/import, frozen init, real cached shard and manifest/mount checks pass.
First draft/stage failures were role-object fingerprint, family feature-schema
and independent attempt-ID declarations. They were corrected before prospective
publication/remote training; failed release001 had no published trajectory or
remote POST. Release002 owns the actual frozen submission; no scientific retry.
Opaque historical retained assets were copied from desktop with SHA checks only
so RML can validate in this fresh owner; no historical model or role execution.
