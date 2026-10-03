# Chemical auxiliary terminal and custody decision, 2026-10-04

Owner: Desktop, `codex/exp/gptrans-chemical-aux`. This decision is prepared on the owner branch; it does not commit, push, archive, or promote the branch.

## Terminal status

- **CPU fixture feasibility:** `T-gptrans-chemical-aux-feasibility` is terminal `INCONCLUSIVE`. The RML package binds only the retained 2026-09-30 feasibility summary. That summary reports fixture checks, but no original raw test-runner receipt or logs are retained; the terminal evidence therefore marks execution unknown and does not assert independently verified test execution. CPU, device, wall, and queue measurements remain missing. No role events, training, model inference, or protected-role access are claimed.
- **Descriptor and fingerprint arms:** both submission_v4 arms have accepted endpoint records and remain `INCONCLUSIVE` for strict V5 qualification. Their paired improvements (2.444253 meV and 1.363584 meV) are below the frozen 3 meV nomination gate. The historical reference is contextual-only, strict prospective reference/comparison qualification is absent, native allocated T4/CPU/queue costs are unknown, and producer logical run IDs differ from the physical Kaggle run ID. See [STATUS.md](STATUS.md) and the per-arm decisions under `training_prospective_v4/`.

No model adoption, scale-up, or official/test evaluation follows. Preserve these qualification gaps as unresolved; a summary-only CPU closure does not validate the full cache or GPU inference equivalence.

## Custody route

Do not adopt the implementation. For eventual branch routing, preserve the complete experiment history in `archive` and integrate only the accepted canonical decisions/evidence and generated RML needed for desktop discovery. Archival is custody, not scientific qualification. Keep the owner branch until that reviewed routing is carried out.

Preserve at minimum `protocol.md`, `feasibility.md`, the original `prospective/trajectory.json`, its decision/policy/cost snapshots, the new summary-only terminal inputs and `prospective/rml_finalized/`, the `training_prospective_v4/` per-arm acceptance/decision/RML packages, `submission_v4/` terminal observations/artifacts, `STATUS.md`, and this decision. Do not replace the original prospective snapshot or synthesize the absent CPU test receipt.
