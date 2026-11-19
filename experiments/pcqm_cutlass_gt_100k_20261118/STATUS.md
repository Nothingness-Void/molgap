# Status

Closed 2026-11-19 JST. Run `pcqm-cutlass-gt-100k-20261118-r2-s42` (`v1`) finished 60/60 passes on both arms and was retrieved. Control `pair_update_norm` is CLOSED. Candidate `cutlass_gt` is NEGATIVE_UNDER_CONTRACT: EMA dev MAE 150.76 vs 139.44 meV (+11.32 meV, gate -3.0). See `decision.md`.

## Earlier run `pcqm-cutlass-gt-100k-20261118-s42` (v1): NO_TRAIN

Superseded by `r2` above. Kernel `nvoid912/molgap-cutlass-gt-100k-20261118-s42` (id 145779324) ended ERROR in preflight, before any training. Read-only log check (2026-11-19): `cutlass_gt` preflight died in its first train step, in `cutlass_gt.py` `_compute_wire_topo_and_pad` at `torch.linalg.inv`, with `CUDA error: no kernel image is available for execution on the device`. The `pair_update_norm` preflight log shows no error line. No metric exists. Outcome: NO_TRAIN (infrastructure). Missing discriminator: none for the science, since `r2` answered the question.

Closure gap: this run has no output manifest, so retrieval stops at `remote_manifests` and run-keyed `accept_run.py` has nothing to read. No shipped command closes such a run, and a hand-written terminal record would be glue. Its two prospective trajectories stay open (ACTIVE). Nothing in the `r2` files changed.
