# Status

Closed 2026-11-19 JST. Run `pcqm-grit-spse-100k-20261118-r2-s42` (`v1`) finished 60/60 passes on both arms and was retrieved. Control `pair_update_norm` is CLOSED. Candidate `grit_spse` is NEGATIVE_UNDER_CONTRACT: EMA dev MAE 153.95 vs 139.44 meV (+14.51 meV, gate -3.0). See `decision.md`.

## Earlier run `pcqm-grit-spse-100k-20261118-s42` (v1): NO_TRAIN

Superseded by `r2` above. Kernel `nvoid912/molgap-grit-spse-100k-20261118-s42` (id 145782179) ended ERROR in preflight, before any training. Read-only log check (2026-11-19): `grit_spse` preflight raised `ValueError: Resume is missing required optimizer moments` (`training/session.py` `validate_session_resume`, called from `load_state_dict`). The `pair_update_norm` preflight log shows no error line. No metric exists. Outcome: NO_TRAIN (infrastructure). Missing discriminator: none for the science, since `r2` answered the question.

Closure gap: this run has no output manifest, so retrieval stops at `remote_manifests` and run-keyed `accept_run.py` has nothing to read. No shipped command closes such a run, and a hand-written terminal record would be glue. Its two prospective trajectories stay open (ACTIVE). Nothing in the `r2` files changed.
