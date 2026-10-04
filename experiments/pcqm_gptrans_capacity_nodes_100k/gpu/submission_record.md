# Four-arm Kaggle2 publication observation

On 2026-10-05, two private T4x2 notebooks were submitted with the owning Kaggle
adapter and passed release reports. Both physical version1 identities were
reconciled against the authenticated remote metadata; the scheduler returned
RUNNING. This observation does not certify remote optimizer preflight or results.

| Physical notebook | Arm | Data scale | Parameters |
|---|---|---|---:|
| node-capacity, ID137072037 v1 | node352 | 100K | 9,678,273 |
| node-capacity, ID137072037 v1 | FFN2 | 100K | 6,822,753 |
| local-scale, ID137072203 v1 | real-bond local channel | 100K | 5,871,201 |
| local-scale, ID137072203 v1 | one live optimizer, EMA9999/EMA999 views | 500K | 5,246,817 |

The frozen ceiling was 10,493,634 parameters. The500K arm used equal optimizer
updates/presentations, not sixty500K passes or two encoder training streams.
Its transfer-study purpose cannot become a STRICT_CAUSAL claim by comparing to
the100K reference. The three100K arms reused the accepted G1+EMA999 reference.

## Independent authorities

- [Node source publication](source_publication_v1.json), [submission receipt](submission_v1.json),
  and [pulled identity observation](remote_identity_v1.json).
- [Local/scale source publication](../../pcqm_gptrans_capacity_relations_100k/gpu/source_publication_v1.json),
  [submission receipt](../../pcqm_gptrans_capacity_relations_100k/gpu/submission_v1.json),
  and [pulled identity observation](../../pcqm_gptrans_capacity_relations_100k/gpu/remote_identity_v1.json).
- Immutable source commit: `f47bb27e14a80bc6115bcfdf0af187f5ba9d327f`.
- Source archives remained immutable. Pulled script bytes differed only because
  the Windows SDK wrote CRLF; raw hashes and exact LF-normalized equality were
  retained. Actual accelerator metadata was `NvidiaTeslaT4` for both notebooks.
- Startup JSON was not yet retrievable during this observation. No bulk fallback,
  inference or protected-role access was performed.

Each notebook reserved at most seven wall hours and fourteen allocated T4 hours,
twenty-eight allocated T4 hours total. Actual measured cost belongs to terminal
acceptance, including idle allocation and failures.

## Monitoring and closure

The [binding](monitor_binding.json) linked both exact runs to the existing Luna B
and controller A. The existing thirty-minute heartbeat was reactivated without
creating another chat/automation or overriding A's model settings. A bounded
monitor tick verified both physical IDs/versions and returned SILENT/RUNNING.
The B conversation subsequently ran the same tick and returned DONT_NOTIFY.
Terminal events require idempotent delivery followed by acknowledgment.

All four prospective trajectories, estimated cost events and policy snapshots
were retained under their owning arm's `rml_plan/`. This is planning evidence,
not completed Replay qualification. Terminal acceptance must retain predictions,
canonical traces, observed roles/cost and decisions. No successor was released.

A local saved-artifact adapter correction designated exactly one canonical trace
for the shared-live500K transaction, while preserving both EMA views and raw
history as hashed auxiliary evidence. Its synthetic retention test passed;
remote training bytes and scientific contracts were not changed by this repair.
