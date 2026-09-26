# Execution state

## Submitted, awaiting startup

At 2026-09-27 04:31 JST both private Kaggle2 v1 notebooks were `QUEUED` with
no startup logs. Requested hardware is not proof of assigned hardware or
successful model preflight.

| Slot | Actual scheduler identity | Arms | Request / returned metadata |
|---|---|---|---|
| dual | `kaseichou/molgap-k1-receiver-and-triplet-s42`, kernel 136015255, v1 | receiver-pair; triplet-aggregate | T4x2 / `NvidiaTeslaT4` |
| rrwp | `kaseichou/molgap-k1-rrwp-pair-s42`, kernel 136015256, v1 | RRWP-pair | P100 / generic `Gpu`; actual device pending |

Release/source identity and the exact downloaded entry/metadata hashes are in
`submission_receipt_v1.json`. Scientific source commit:
`f2d794f953af437af3fab60b33d49720531c7195`; source archive SHA256:
`5a0e5f29b836d4e92dc7b52937f4ca05521777d434f9314de548d5be0120297f`.
The source dataset was private and ready; all five bootstrap files including
the unexpanded `source_payload.bin` were verified remotely before submission.

Kaggle generated the dual notebook slug from its title, adding `and`.
The frozen source's embedded trace/run identifier remains the predeclared
logical identity without `and`. The receipt binds that identity to actual
kernel **136015255/v1** with matching source and entry; nothing was resubmitted
or rewritten. Terminal provenance must retain this explicit alias and receipt,
not invent a second physical run. Monitoring uses only the actual slug.

## Validation and monitoring

- New static/mock checks: 18 passed; no local model construction or inference.
- V5 targeted checks passed before release; prospective RML `validate`,
  `rebuild`, and `check --frozen` passed after all four plans were created.
- Three training trajectories plus one conditional separate `NO_TRAIN` audit
  plan exist. No terminal result or new replay-ready training entry is claimed.
- `monitor_binding.json` binds two independent local control stores to the
  existing Luna B `01a04479-ca44-7d31-95c4-6be485f256cc` and existing A
  `01a025a1-3b87-7781-8a91-f183193f7865`.
- The existing heartbeat `molgap-k1-conjugated-dual-kaggle2-monitor` was updated
  in place, enabled every 30 minutes, and renamed for this study. Healthy
  checks are silent. Each terminal/fault produces one idempotent handoff to A
  without overriding A's model or reasoning setting. The first terminal job
  does not close monitoring for the other job.
- Training acceptance uses `accept.py` independently for each slot. A owns
  scientific attribution, strict RML terminal closure and replay rebuild.
  Only after accepted training may A release the separate frozen500K internal
  development inference audit. There is no automatic scale-up, extra seed,
  training successor, or protected-role access.
