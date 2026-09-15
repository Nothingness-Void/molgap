# Terminal failure diagnosis — 2026-09-15

## Classification

ERROR — infrastructure/resource-selection mismatch before candidate execution.

This run is not a scientific failure and supplies no model metric. The
version-2 kernel requested the NvidiaTeslaP100 machine shape, but Kaggle
exposed one Tesla T4. The candidate entry point intentionally stopped at its
accelerator guard:

    RuntimeError: Candidate requires one P100, got Tesla T4

The terminal cause is therefore an unmet hardware allocation contract, not an
architecture, data, optimizer, numerical-stability, or checkpoint failure.

## Evidence

- Kernel: kaseichou/molgap-k1-edge-conditioned-slot-s42, version 2.
- Durable handoff:
  platforms/_records/kaggle/training/k1_edge_conditioned_slot_s42_v2/terminal_handoff.json.
- Terminal log:
  platforms/_records/kaggle/training/k1_edge_conditioned_slot_s42_v2/molgap-k1-edge-conditioned-slot-s42.log.
- Downloaded source tree:
  platforms/_records/kaggle/training/k1_edge_conditioned_slot_s42_v2/_molgap_source.
- Retry identity: source commit
  29cb2b9a6144b1f3badc70035a16ca6df0843333; source archive SHA-256
  6e16f4011f860cc4e31615d5a936c410f632e4ae9f55f4fbe98d884a8ae84a65.
- Download completed with 156 files and 2,288,120 bytes. No candidate output
  directory was produced.

The log reaches the launcher guard after dependency installation and fails
before candidate import/training, data access, epoch logging, checkpointing,
or scientific acceptance. The package-install dependency warnings are
incidental and are not the terminal cause. The acceptance was correctly not
run; no model inference was executed, and official, shadow, and test roles
were not read.

## Decision

Classify version 2 as infrastructure evidence only. It does not provide a
scientific result, does not close or advance the edge-conditioned-slot
architecture question, and does not alter the frozen K1 handoff. Preserve the
downloaded log, source tree, and terminal marker unchanged. No repair, retry,
successor, seed, scale bridge, or official/test evaluation is authorized by
this diagnosis.
