# V1 local source-boundary failure

Parent-reported2026-10-26: all three v1 prospective records were published.
BN native preflight decoded the training role, then failed before GPU compute
or any gradient update because the in-process local
adapter imported `molgap.local_training`, excluded by `runtime_source_files`;
the loaded-source guard correctly rejected the unbound module. Formal training
updates:0. There is no formal training trace or successful native proof; retain
the failed invocation and its log.
The other two v1 arms were not launched. Terminal disposition remains unresolved;
do not auto-finalize synthetic terminal evidence. This is infrastructure
failure, not a scientific result; no candidate mechanism or recipe is changed.

Retain every v1 Spec, workflow, plan input, prospective, receipt and preflight
artifact at its original path. Do not reuse or rewrite v1 output directories.
V2 adds only explicit `src/molgap/local_training.py` source inclusion, uses
`attempt_2/<arm>/` envelopes/prospectives, `-r2` logical run IDs and distinct
trajectory IDs. V2 reuses the exact frozen v1 recipe files and original init.
Each v2 proposal points to its corresponding original v1 trajectory as a prior,
not inherited training or evidence of a completed scientific result.
All training horizons, runtime, roles and candidate defaults remain unchanged.
Comparisons remain PENDING; no strict reference or successful training is inferred.

After the parent commits this helper and note, generate envelopes against that
frozen HEAD (do not commit generated envelopes into a later source HEAD):

```powershell
& 'D:/文档/molgap/.venv/Scripts/python.exe' experiments/pcqm_k1_selective_ffn_norm_100k/prepare_inputs.py --attempt v2
& 'D:/文档/molgap/.venv/Scripts/python.exe' experiments/pcqm_k1_selective_ffn_norm_100k/prepare_inputs.py --attempt v2 --prepare bn_reference
```

Prepare `ffn_rms` and `local_residual_gain` separately with the same attempt flag.
Preparation retains the existing owner, source/recipe checks and prospective
publication before execution. No core change, training or commit is performed
by this repair task. This note records parent observations, not an independent
GPU execution or artifact acceptance by the declaration agent.
