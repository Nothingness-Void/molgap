# Stage 6 infrastructure failure

Kaggle kernel `nothingnessvoid/molgap-500k-v4-full-gps-final-epoch` version 7
failed before training with `Resume scientific/source identity mismatch`.
The resume archive itself passed its SHA256 and cursor checks, and the runtime
certificate was accepted. No epoch ran and no new checkpoint was emitted.

The stage-5 wrapper had legitimately required source identity
`811a40969ece2df05d95556901fcc7c3ebcaa6f46c95e19a4e9fa2e856fad3ae`
to consume the epoch-16 checkpoint. After that transition, every stage-5
checkpoint was written with current source identity
`4218940805a84dbf69971452e3c113c7f5e8281df8be62371836b8bc3ff2b649`.
The one-epoch wrapper incorrectly retained the prior transition identity.

The repaired wrapper therefore leaves `resume_source_sha` unset, making the
runner require the current source identity already stored in the accepted
epoch-59 checkpoint. This is an infrastructure-only provenance correction; no
scientific field, artifact, data, RNG, optimizer, schedule or exposure changes.

- Failure JSON SHA256: `4232d185c794ae56fdacb9ff1145585484b3c21293a741fc53535321968397bb`.
- Terminal log SHA256: `5836858d075569f0bef9d72c91acb77efac3242a59cf9567971963d8d5cc0a58`.
- Raw evidence: `platforms/_records/kaggle/training/pcqm_500k_v4_stage6/full-gps-final-epoch-v7/`.
