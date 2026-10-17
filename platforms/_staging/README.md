# Platform Staging

Generated work area for package snapshots, frozen inputs, preflight outputs, and staged platform artifacts.

This is not a source of truth. Keep an active release immutable while it is in
use. After reconciliation, a referenced release can leave local staging once
its exact source/configuration and required artifacts have verified durable
custody. A reference requires recoverability, not permanent duplicate storage.

- Platform workflow: [platforms README](../README.md).
- Retrieved durable evidence: [_records README](../_records/README.md).
- Retire verified redundant staging under the [checkout lifetime rule](../../BRANCHES.md#local-checkout-and-artifact-lifetime); preserve unresolved attempts and sole evidence.
