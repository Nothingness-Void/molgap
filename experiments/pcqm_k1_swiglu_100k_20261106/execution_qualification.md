# Execution qualification — 2026-11-06

The user released necessary qualification and the planned fixed100K candidate
training. There are two scientific comparison arms: retained original K1 A and
new SwiGLU B. No reference is retrained.

Existing `submit_workflow.py --experiment ...` without platform execution returned
PREPARED_FOR_PLATFORM in127.802802s and published the canonical prospective record.
Frozen source `b43b7fc70320caf4ca588620d76b7f0ea2a10f70`;
Spec `af3b8b526cd10056375afc0e59b878c780599c709b06bdedb64cf7ce62b7f2fe`;
package `acc69d83e9d1b15f7c0149506d4c8c881f77833ae7ef4bd38c0c8cf4488fb96d`.

## First existing sandbox call

Run-keyed sandbox executed once after prospective publication. Exit1:
`SandboxCheckFailed`, `addon_taps_fire`, `attached=0`, `silent=[]`.
The sandbox as a whole is not passed.

The same report records PASSED step0 bitwise evaluation/training, nine mutation
gradient groups (`swiglu_ffn_0` through `_8`), and fresh-load state roundtrip.
Each FFN has four connected nonzero trainable parameter tensors with no missing,
zero or nonfinite gradients. Reloaded predictions are reported bitwise equal.
These are the original checker's flags, not independent proof of every assertion.
The infrastructure owner subsequently found that step0 used the candidate as
its own comparator, and the old reload check did not actually perturb replacement
parameters despite reporting an output-change flag. Thus the first report does
not prove original-model equivalence or restoration after real parameter movement.
Corrected qualification remains pending. No native T4 result follows from these CPU checks.

Raw existing-command stdout:
`D:/molgap-artifacts/reviews/k1-swiglu-sandbox-20261106.json`.
It contains the launcher interpreter JSON line followed by the sandbox result.
No new test runner, fake tap, modified validator or remote submission was used.
The exact error and involved paths were reported to the infrastructure owner
chat `01a14421-e9d1-73a1-9059-3723d77d0f02`, which was asked to repair it.
Remote publication/submission is held pending that qualification disposition.

## Corrected qualification and submission

The infrastructure owner fixed the existing sandbox on desktop commit986db6ea
and applied it to this worktree as c4da3a75. The new verifier observes registered
replacement modules, labels self-comparison explicitly, and actually perturbs
replacement parameters during save/reload validation. No dummy tap, new test
framework or scientific-model alteration was introduced.

Actual report `D:/molgap-artifacts/reviews/k1-swiglu-sandbox-fixed-20261106.json`
was inspected: status PASSED; addon_taps_fire attached9/silent[]; nine groups of
four parameter tensors have connected nonzero finite gradients; output changes
after real parameter movement, and saved/fresh-loaded predictions are bitwise
equal. `step0_reference` is `same_recipe`: step0 checks repeat construction of
the candidate, not equivalence to the original K1 function.

The existing prepared-run submit command reused the original frozen source,
Spec, prospective records and package. It returned SUBMISSION_BOUND for Kaggle3
job142921674/version1. No baseline training, replacement source freeze, duplicate
submission or additional model diagnostic was performed. Scheduler state and
receipt paths are owned by STATUS.md. Native T4 repeat/resume and scientific
acceptance remain pending, not inferred from CPU sandbox or RUNNING.
