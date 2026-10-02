# Desktop infrastructure review - 2026-10-02

The audit found the registered local experiment infrastructure reasonable for
selective server reuse after bounded implementation and compatibility repairs.
It did not establish universal family/platform support or real accelerator
qualification. The user authorized local synchronization after review.

## Source and integration scope

- Inspected Desktop tip: `f4956019453b75d85347038d46a917fddc9dbdd1`.
- Primary additive infrastructure source:
  `9dbeb26abf284f5232aba8a862a38c993d0150b8`.
- Prior family output and epoch/metric corrections were reviewed at
  `2e2f33c749cf5331c1b5ca7b3ed5f22de3c7574b` and
  `757519915db015c1219148581eb6797c3a9a4765`; the server already retained its
  corresponding output hooks and stronger independent metric definitions.
- Server integration base: `cea60f1c54f0e295bc53ddce551d3d64967f33ac`.

The additive integration covered Spec v2 K1 registration, static family/addon
execution, reviewed source inventory, K1 and GPTrans owning screen adapters,
isolated assigned-device workers, all-arm preflight, Kaggle staging and final
release binding, scoped output costs, incomplete terminal translation, and
all-arm planning validation before any publication.

The server retained `prepare-release`, `check-preparation`, EdgeState output and
training hooks, author/input arm registrations, its CPU platform adapter,
server ownership defaults, exact frozen prospective attempt IDs, and strict
replay closure. Historical model/GPTrans runner bytes, scientific contracts,
production registry, Desktop experiment records and remote jobs were preserved.
The legacy lifecycle tests were retained in a separate test module rather than
discarded during the workflow-file conflict resolution.

Use the [registered workflow](REGISTERED_EXPERIMENT_WORKFLOW.md) for its exact
supported families/modes. Existing server preparation and scientific/replay
release gates remain in [the server workflow](EXPERIMENT_WORKFLOW.md).

## Reproduced defects and repairs

1. GPTrans's static validator accepted a `centered_logits` recipe for a
   reference arm. Static and runtime checks now reject a recipe mode that
   differs from the executable arm. The synthetic regression failed on the
   inspected Desktop source and passed after repair.
2. If a training worker failed to spawn, its retained terminal status stayed
   `running`, blocking incomplete terminal translation. The spawn exception
   now records the failed arm and exception type before cleanup. The synthetic
   second-worker spawn regression likewise failed before repair and passed after.
3. Incomplete translation's version-derived attempt name was incompatible with
   server prospective identity. The CLI now reads the exact owning action under
   an explicit repository root. The library retains unknown identity when no
   prospective binding is supplied; it does not invent an attempt from a version.

Compatibility work imported hash/loading helpers from their actual shared
owner without changing the historical GPTrans runner. The source inventory
included server staging and diagnostic-binding dependencies. New recipe builders
pinned full metric semantics. EMA-only development observations were supported
while exact recipe/trace, Gap/eV/MAE, weights and declared-role checks remained.

## Executed verification

All commands used the project's `.venv/Scripts/python.exe` and the selected
checkout's `src` on `PYTHONPATH`. Fixtures used synthetic tensors, local temporary
repositories and mocked platform transport/processes. No molecular evaluation
role, benchmark training, remote job operation or submission was performed.

| Verification | Result |
|---|---|
| Inspected Desktop infrastructure suite | 766 passed, 7 skipped |
| Adapted workflow, normalization and initial defect regressions | 42 passed |
| Combined new and retained server lifecycle, source/release, receipt/terminal, adapter, V5/comparison and planning regression | 1166 passed, 9 skipped |
| Supplemental final identity/metric/defect and CLI checks | 18 passed, 65 deselected |
| Existing local server RML with original and integrated code | Both validated 68 trajectories / 69 evidence envelopes; frozen indexes passed |
| Diff whitespace and merge-marker checks | Passed |

These overlapping selections are not additive test counts. Skips were retained,
not represented as executed checks. The existing package deprecation warning
did not affect the results. No RML rebuild or canonical/derived evidence edit
was required by this infrastructure integration.

## Qualification limits

A fresh Desktop Git checkout could not run the complete frozen/portable RML
check because the independently required retained artifact
`platforms/_records/kaggle/training/pcqm_gptrans_local_inductive_bias_100k_s42_v2/attempt_002/arms/rwse16_local_edge/output/best_model.pt`
was absent. Git intentionally does not carry large model assets. The check stayed
blocked; no evidence was fabricated or validator relaxed to make it pass.
This limits independent artifact verification from Git alone and does not
invalidate the focused source/interface tests above.

The registered preparation backend covers Kaggle T4x2, with explicit K1 v2 and
retained GPTrans modes. IMS/SCNet operations and arbitrary future addons retain
their owning adapters. Real cache/hardware, executable resume, native total cost,
scientific comparison and replay qualification remain separate gates.
