# K1 bounded local mixing and clean fingerprint supervision

Desktop owner: `codex/exp/k1-local-mixing-clean-aux`, based on verified desktop
316680b0050677bb7887d603887771d0de5a6b61. Read [STATUS](STATUS.md) for release
state and [protocol](protocol.md) for the two independent scientific deltas.

This pair contains two candidates. The existing K1 V4 reference is reused;
no third training arm or reference retraining is authorized. Source reuse and
imports are pinned in [reuse_provenance](reuse_provenance.json).

The owning V4 loader, Python sampler and epoch loop were extracted from archive
8821b5ce893680121260627436dc11a8f7fd8403 into `molgap.k1_screen_training`.
Output events use `FamilyOutputSession`, prospective records use
`experiment_cli plan-prospective`, and comparison checks use
`comparison_readiness`. Packaging, release checking, receipts and platform push
remain with their existing owners. No new RML framework is included.

The new family version 2 describes the 100K contract; version 1 continues to
describe full training. The two addons and the output profile are explicit.
Current implementation is preparation, not an accepted remote trainer.
Runtime qualification and platform wiring are gated by reference availability.
