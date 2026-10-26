# Kaggle v1 failure inspection

2026-10-26 JST. Both exact Kaggle1 version1 jobs are ERROR: single140826099,
mean2140826274. The question remains unresolved; no successor was submitted.
[Machine-readable observations and file hashes](inspection.json) bind the
selectively retrieved diagnostics under D:/molgap-artifacts/reconciliation/.
No weights, prediction payloads or native-proof ZIPs were downloaded for this
inspection. Scheduler reconciliation was performed by Luna.
All four retained runtime-provenance contexts match their frozen package source
commit/archive/package and Spec/run/arm IDs in a local JSON identity check.

## Confirmed failure

Both native all-arm preflights completed, and all four training workers started.
Every retained train.log has the same traceback:

`family_training_outputs.py:248 -> create_resume_bundle ->
experiment_resume.py:405 -> _json_bytes -> json.dumps ->
TypeError: Object of type Tensor is not JSON serializable`.

The frozen64933302 source's `_native_cursor` copies the entire checkpoint cursor
into manifest metadata. Clean-BN state stores tensor-bearing `weight_state` in
that cursor. The first10-pass resume-chunk export reaches JSON serialization of
those tensors. Tensor state belongs in the existing checkpoint payload; a JSON
manifest needs only the supported cursor metadata. This is an infrastructure
failure, not evidence against pretraining or either forward recipe.

The real preflight passed without reaching the scheduled10-pass export. Its
success did not establish that this Clean-EMA/chunk combination could be saved.
The incident and exact traceback were sent to the infrastructure owner for a
minimal repair of the existing transport, with bounded existing-workflow
verification rather than another training framework or expanded default suite.

## Actual progress and provisional metrics

Every canonical trace retains10 completed observations,39060 optimizer steps,
4999680 presentations, and a checkpoint digest at pass10. The visible logger and
training_progress.json stop at9 because chunk export failed before the next log.
Checkpoint bytes have not been downloaded or validated: the trace and source
ordering support a retained checkpoint record, not a verified resume guarantee.

| Pair | Scratch pass10 Clean-EMA MAE eV | Pretrained pass10 Clean-EMA MAE eV |
| --- | --- | --- |
| Single | .114605255425 | .113491639495 |
| Mean2 | .111328274012 | .110644631088 |

These are incomplete, same-pass diagnostics, not the frozen30-pass selected
endpoint, scientific acceptance or promotion. They cannot decide pretraining
benefit at the planned endpoint or establish a final K1 limit.

## Cost and recovery boundary

Root allocation ledgers retain single6545.841682445 wall seconds and
13091.683364890 T4 device seconds; mean210571.080998975 wall seconds and
21142.161997950 T4 device seconds. Total observed allocation is9.509401490 T4
device-hours. Provisioning before the recorded scope and queue are not inferred
as zero. Retained runtime/provenance/source bindings remain separately checked;
this inspection does not certify the native proof ZIPs or completed replay.

Four arms are incomplete and cannot be called replay-ready. Before continuation,
repair the metadata export, verify actual checkpoint/source/exposure identity,
and preserve the original optimizer/scheduler/EMA/RNG and Clean-BN state through
the owning recovery path. Do not start again from initialization merely because
the portable chunk manifest failed. No continuation is launched by this record.
