# Desktop modular runtime acceptance

Date: 2026-10-15. Owner: `molgap-desktop`. Scope: infrastructure only.
This records integration, not model promotion or native qualification.

## Provenance and scope

- Desktop base: `1be9efb7ed6785ac477281d95f6beae651947d6f`.
- Server reviewed tip: `ded33bea2c2a840ba9569ab9dd93b7b9bb7dde7a`.
- Four-layer source: `938ddc1124899e2df78ece38401e9eabc9cd9c31`.
- Generic ABI prerequisite: `43e5fc7b1cb365c6baf09b5d7150bdaca4ab7523`.

Selective shared-code port, not a server merge. The
[runtime guide](MODULAR_EXPERIMENT_RUNTIME.md) owns interfaces and limits.
Desktop accelerated defaults, scientific modes, seed43 compatibility, target
bindings, platform boundaries, V5 invariants and historical packages are retained.
No server controller, campaign transport, research records or queue is imported.

Desktop bridges add typed addon/config registration, compatible graph factory
dispatch, owner checkpoint state, safe CPU loading and source closure. Config
builders reuse Spec validation; unsupported preparation returns no route.
Synthetic fixtures use desktop source/attempt identity rules without weakening
production gates. IO source allowances retain sensitive-directory, suffix,
case-alias, committed-byte and credential rejection.

The default RML trace manifest now reports checkpoint identity only when the
retained trace observes it. Synthesized manifests remain replay-ineligible;
no canonical historical evidence or scientific disposition is rewritten.

## Executed CPU verification

Independent executor: `gpt-6-luna`, `max`; project `.venv` Python, isolated
integration checkout with `PYTHONPATH=src`. CUDA/HIP/ROCR masks use `-1`;
CUDA unavailable and visible device count zero were asserted before execution.

| Final test group | Result | Seconds |
|---|---|---|
| New module tests; source package, execution, Spec, CLI; runtime navigation link | 398 passed, 1 skipped | 134.45 |
| `tests/test_terminal_trace_closure.py` | 40 passed | 13.86 |
| Full synthetic package/addon/train/checkpoint-resume/acceptance/RML closure, repeated separately | 1 passed; included in the first group | 34.84 |

Earlier compatibility run: 565 passed, 4 failed, 1 skipped. The four stale
package-boundary cases were repaired and retested in the final group above;
the entire long compatibility group was not rerun. Do not sum repeated cases.
The one skip is the host's inability to create a symlink, not a missing model
artifact. Final `compileall -q src/molgap tests` and `git diff --check` passed.
Final navigation/layout run: 105 passed, the ten unchanged baseline failures
below, no newly introduced failure; runtime/receipt links and entrypoints passed.

## Baseline exclusions

The other ten documentation/layout failures reproduce on the unchanged primary
desktop HEAD above. They concern routing-document line budgets, experiment
README/index/decision pointers, frozen snapshot links and old naming/root-depth
checks. This port does not claim a clean full-repository layout acceptance or
rewrite frozen evidence to remove those failures.

GPU/DCU, real molecular caches/checkpoints, protected roles, remote submission,
native cost/speed, scientific comparison and promotion were not exercised.
Synthetic CPU success does not release compute. Clean EMA remains an opt-in
component requiring its own calibration, selection and checkpoint contract.
