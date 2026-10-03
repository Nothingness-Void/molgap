# Submission preparation review, 2026-09-30

The attempted release stopped at the CPU label gate in `cache_decision.md`.
There is no Kaggle launch receipt or trained chemical arm.

## Reused components

- Accepted V4 archive/graph validation and official split membership metadata.
- `ChemicalLabelEncoder`, `build_cache` and the fail-closed cache loader.
- Existing GPTrans V4 training, optimizer, EMA and checkpoint operations.
- `RunContext`, `FamilyOutputSession` and canonical trace/event recording.
- RML prospective planning, terminal finalizer, rebuild and frozen check.

## Narrow additions

- Export authenticated train-only SMILES in compact JSONL.
- Optional epoch, selected EMA predictions and resumable-state events from the
  existing trainer into the shared output session.
- Thin owning CLI binding of Spec/package/contract and training source identity.
- A remote-only short optimizer-step profile. It does not measure CPU cache
  attachment, loader throughput or end-to-end wall overhead and does not qualify
  the frozen wall-time gate. No actual CUDA profile was executed.

## Checks actually executed

The original targeted synthetic pass had 72 passing checks. Four train-export
regressions subsequently passed, including the JSONL serialization repair.
Luna then ran the focused profile/event/session set: 14 passed, covering source
commit and archive mismatch rejection before training. These sets overlap;
their counts must not be added as a unique-suite total. No real GPTrans model
was constructed or run locally. CLI help, Python compilation, whitespace review
and `research_memory check --frozen` passed.

Actual CPU work verified the complete train export and independently reproduced
the failed chemical row. The stopped label build is incomplete. Its retained
process interval is a measured lower bound, not total submission-preparation
cost; orchestration and the first failed export have no complete cost ledger.

## Remaining capability boundaries

No Spec v2 training pair, release package or remote receipt was frozen after
the cache failure. The standing two-new-direction preference led to preparing
descriptor-only and fingerprint-only as the prospective pair; the older
baseline-plus-joint draft was not executed or retrospectively rewritten.

The shared comparison purpose registry still needs a reviewed way to express
the auxiliary loss intervention; an architecture-only comparison cannot hide
a changed loss identity. Strict retained/same-run reference binding and actual
T4 equivalence, resume, throughput and wall-cost qualifications remain required
before release. Historical V4 evidence must not be relabeled replay-complete.

This attempt verifies CPU failure capture and RML closure, plus synthetic
producer event wiring. It does not validate the full submission-to-replay path.
