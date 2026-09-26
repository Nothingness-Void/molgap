# Linear attention screen status

Closed, no promoted candidate and no active job in this chain.
[Scientific decision and attribution](decision.md).

- The original v1 training stage completed and is strictly accepted, with a
  complete candidate/reference training replay entry and native measured cost.
- Its separate audit failed and is finalized as `INFRASTRUCTURE_ONLY` under
  the original physical run ID. [Failure diagnosis](results/audit_failure_v1.md).
- The frozen-audit recovery v1 completed and passed independent no-inference
  acceptance. It is separate `retrospective_partial` / `NO_TRAIN` evidence,
  not a fabricated prospective continuation under the original run ID.
  [Recovery receipt](results/audit_recovery_receipt.json).

[RML closure and verification](results/rml_closure.json).
Original prospective files, models, checkpoints, raw logs and prediction chunks
are retained. Targeted tests: 63 passed; RML validate/rebuild/frozen check passed.
No local model training/inference, new seed, 500K training, full training or
protected-role access. Missing audit device/setup/queue costs remain missing.

The existing Luna B heartbeat is PAUSED after one terminal handoff. No new chat,
automation, retraining, or architecture successor was created.
