---
name: ims-molgap-workloads
description: Submit, resume, reconcile, or retrieve MolGap CPU and A100 jobs on the IMS/RCCS molecular research server.
---

# IMS MolGap Workloads

Read `platforms/REMOTE_HANDOFF.md` in the owning checkout before **any** IMS
command, including read-only discovery. It owns the authorized path and
scheduler boundary. Keep credentials outside Git and logs.

## Submit a committed V4 fixed-PCQM run

Commit the experiment run spec and clean source first. Use the existing V4 spec
and authority for the model, fixed data, contract, runtime, resources, entrypoints
and required outputs. Set the path to the key once with MOLGAP_IMS_SSH_KEY (or
pass --key). From the owning checkout, run the authorized submission:

    & $python platforms/ims/v4_submit/submit_from_windows.py --spec $spec

Use D:/文档/molgap/.venv/Scripts/python.exe as $python. The helper discovers the
checkout and stores the payload in ~/molgap-ims-payloads/<run_id> unless
MOLGAP_IMS_PAYLOAD_ROOT or --payload selects another location. A retained
--payload can replace --spec for later operations, including after HEAD moves.
Use --status for durable records. A submit without --status or --resume can
queue jobs immediately. Connection overrides are --host, --port and --user.

The adapter checks clean committed source, archive/contract/data identities,
V4 batch/precision restrictions, duplicate run directories, maintenance and the
preflight dependency. Use its result rather than repeating those checks by hand.
Read `platforms/ims/v4_submit/README.md` for that route's schema or failures.
Its fixed 100K/500K datasets are the accepted, byte-identical store; never rebuild them.
Other workloads use their owning `platforms/ims/` adapter; the shared editable
experiment-input frontend does not submit IMS jobs.

## Queue and resource rules

- Group related, authorized items in the serial single-job mode when suitable.
  It runs one training job. Single-recipe V4 keeps an afterok preflight.
- Finish every check **locally before queueing**: run the whole chain a few steps
  on the local machine, including resume after a kill and the early-exit checks.
  Inside the job keep only cheap data and identity checks at the start (exit 3).
  Never use the queued job to find code errors.
- Every A100 has **80 GB of GPU memory**. The local GPU has 8 GB, so local runs
  check logic and equivalence only; size capacity for 80 GB.
- Host RAM is bound to cores: 1.875 GB per core, at most 16 cores per GPU
  (30 GB per card). Source: RCCS manual, jsub page. There is no `mem=` option.
  Prefer GPU memory over more cores: keep datasets resident on the GPU as compact
  contiguous tensors plus offsets, load shards as a stream, and take batches by
  index on the device. Do not buy cores or cards for host RAM.
- Look up IMS limits yourself (this file, `platforms/REMOTE_HANDOFF.md`, the RCCS
  manual, read-only `jobinfo`/`showlim`). Do not ask the user for resource values.
- For serial-chain spec and local rehearsal, use the committed-worktree
  instructions under Serial chain in platforms/ims/v4_submit/README.md.

## No hand-written glue

A test that runs exactly once may be hand-written. Anything that will repeat in
later submissions must not be.
Anything that repeats (spec generation, local rehearsal, gate/skip/pause/resume
checks, packaging, submit, retrieve) is an infrastructure command. If a command
is missing, extend the infrastructure first, then run it. A new throwaway script
in an agent report is a defect: fold it into the infrastructure.

## Reconcile an existing attempt

Identify the owner, run and attempt before connecting. In the authorized remote
session, inspect the live account:

```bash
jobinfo -c
```

With the original --spec or --payload, add --status to see the bound
submission journal, prior job IDs and worker stages. The live scheduler is
authoritative for queue state. Repeat submit on a bound run to get the existing
journal. Unknown submission states require reconciliation by exact identity.
When the scheduler has released the exact paused predecessor and continuation
is authorized, add --resume. This reuses the staged source and contract and
sends only the spec and launcher.

Add --retrieve-to DIR to the original --spec or --payload to fetch completed
V4 outputs with worker digests. Use the owning experiment's retrieval entry
for older runs without worker digests (pass `--to DIR/training`; scorers read `DIR/training/<item>`), and its
acceptance entry for closeout. `run.py --script` also runs committed `experiments/<name>/*.py` scripts.

## Scientific and hardware boundaries

The repository owns branch, role and resource authority. A new question needs
prior RML review and a prospective trajectory before diagnostics or training.
Use CPU for graph construction and acceptance before A100 training. Follow the
owning artifact acceptance and terminal workflow; scheduler completion is not
scientific acceptance. Keep native CPU/A100 costs and missing evidence explicit.
Read `platforms/V4_EXECUTION_CONTRACT.md` when changing V4 bindings, and
`research_memory/LIFECYCLE.md` when closing terminal evidence.
