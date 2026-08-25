# IMS Experiment-Record Snapshot

Captured: 2026-08-25 13:32 JST

This read-only snapshot preserves experiment evidence from
`sm2@ccfep.ims.ac.jp:/lustre/home/users/sm2/chou` for these project roots:

- `molgap`
- `molgap-gpu-smoke`
- `molgap-pcqm-route-b`
- `molgap-phase8-secondary`
- `molgap-repaired-2m-3d`
- `molgap-top20-qm9`

## Inclusion contract

The snapshot contains logs, metrics, manifests, decision/readme text, CSV/NPZ
result records, result figures, checksum files, Python experiment sources,
shell scripts, and PBS adapters. It excludes Git metadata, environments,
vendored dependencies,
inputs, raw data, graph shards, caches, fusion inputs, models, checkpoints, and
the large split CSV. The excluded assets remain identified by hashes in the
retained manifests and experiment records where those contracts existed.

## Verification

- Files: 1,272.
- Remote-to-local SHA-256 verification: 1,272 passed, 0 missing, 0 mismatched.
- `remote_records.tar.gz` SHA-256:
  `1e4ba6c342f2d05b975ec86e9607e292a8f2dd2873c11c25ef86d80f9df01e92`.
- `remote_sha256.txt` is the server-computed digest list.
- `local_sha256.txt` is the independently recomputed local digest list.
- `remote_inventory.tsv` records remote size, modification time, and path.
- `remote_figures.tar.gz` is the supplemental result-figure archive; its
  SHA-256 is
  `3a897fc1fc462916999f069ab232ad4927d08701cbc5e319397bb18aee382fe5`
  and its 53 files are verified by the corresponding figure hash lists.
- `remote_tree/` preserves each path relative to the remote project parent.

The login host did not expose the scheduler client in the non-interactive PATH;
that failed read-only query is retained in `scheduler_snapshot.txt`. Scheduler
job provenance remains available in the downloaded submission records and logs.
