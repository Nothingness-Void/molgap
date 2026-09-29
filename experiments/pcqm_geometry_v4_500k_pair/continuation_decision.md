# Geometry V4 500K dual-arm checkpoint continuation, 2026-09-29

The user authorized completing both original Kaggle1 arms after the frozen
attempt-001 cost stop. That earlier `STOP_FOR_COST` decision remains true under
its original 12/16 T4-hour limits. It is neither reversed nor relabeled as a
scientific result.

The original remote stage preserved five epochs and resumable state for both
arms. The four required binary files per arm were retrieved individually and
their SHA256 values matched the original remote stage manifests. The observed
epoch-five 60-epoch cost projections were 13.18 and 20.89 T4 device hours;
the separately authorized continuation caps are 16 and 26 hours, respectively.
The same 60-epoch scientific recipe and 39600-second stage boundary apply.
The exact budget, source/data identity, role seals, and acceptance requirement
are frozen in `continuation_contract.json`.

The new work uses a private Kaggle1 checkpoint dataset and the existing V4
trainer's resume entrypoint. Every stage must verify the checkpoint manifest,
file hashes, arm, epoch cursor, old source archive identity, scientific contract,
runtime software fingerprint, and GPU identity before advancing. This decision
authorizes sequential bounded stages until both arms reach 60 epochs or a new
measured blocker intervenes. It does not authorize a fresh restart or protected
role access. The two new prospective RML trajectories own continuation results;
they will be closed independently after acceptance.
