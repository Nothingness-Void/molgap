# Exact-2M Hard20K Rescue

Status: training candidate; the future sealed 20K remains unopened.

This experiment appends 20,000 leakage-safe hard-region molecules to the exact
2M coverage model, then warm-starts GPS7, GPS9, and the dual-GPS head. The old
2M prefix receives sampling weight 0.09 and the appended suffix weight 1.0,
which makes the suffix about 10% of expected training draws without discarding
the broad base distribution.

The selection excludes exact identities from the existing 2M data and excludes
all scaffolds present in common/OOD/P8-hard, PCQM4Mv2 proxy, prior sealed 10K,
and future sealed 20K. See `selection_report.json` and `selection_audit.csv`.

Development pass criteria:

- P8-hard average and Gap MAE recover at least 0.0015 and 0.0020 eV versus the
  incumbent comparison used by the exact-2M gate.
- OOD and PCQM4Mv2 do not show a material regression.
- No production/default registration occurs before the development gate passes.

The exact-2M and 2.02M monolithic graph caches are archived after the run to
the SCNet 400 GB `work1` share. Their original project paths become symlinks
only after byte count and SHA-256 verification, so later jobs do not rebuild or
need path changes.
