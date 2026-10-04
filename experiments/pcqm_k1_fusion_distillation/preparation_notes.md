# Preparation findings — 2026-10-04

Reused: current K1 screen trainer, recipe builder, typed Spec registry, shared source inventory, prepare-workflow, per-arm prospective RML, Kaggle accelerator submitter, launch reconciliation and terminal output acceptance. No baseline was trained.

Added capability: one detached fixed-teacher MSE hook and a strict train-only cache loader. Both weights are typed addons; identity, initialization, sampler, trace, selected/resume states, native costs and role handling remain with their existing owners. One concentrated CPU regression batch passed 167 tests; the subsequently discovered objective-purpose gap required one additional passing regression.

Concrete impediments:

- Installed Kaggle SDK preferred the globally logged-in Kaggle1 OAuth session over the explicit Kaggle3 key. The small platform credential helper binds the requested account; authenticated own-kernel listing confirmed `nvoid912` before publication.
- Comparison prelaunch lacked a registered training-objective purpose. The existing validator now permits only the declared loss identity intervention; architecture differences still fail for this purpose.
- Fresh worktree lacked four pre-existing acceptance artifact files. Exact accepted hashes were restored from desktop caches, without retrieval or retraining.
- The first plan incorrectly used a physical reference run ID where RML required an evidence ID. Static source/import/recipe checks passed but planning rejected it before publication of student records or submission. The second complete preparation used the accepted reference evidence ID; no remote retry occurred.
- Transport staging was initially placed inside `experiments`, where copied trajectories would enter RML discovery. After the accepted push, both unchanged staging directories were retained under the ignored platform source-package cache. Canonical student trajectories remain in their owning experiment. Future prepare-workflow output should start under platform staging or a local directory outside experiment discovery.
- Kaggle source pull converted the entry script's LF to CRLF. Raw hashes and exact LF-normalized equality are both recorded; executable source and frozen launch digest matched. Payload/archive hashes remain exact-byte bindings.

Measured local teacher inference: 66.5814 assigned RTX5060 device seconds, 66.7407 body wall seconds. The concentrated test batch took 37.82 seconds. Other stage totals were not instrumented and are not claimed. Remote T4 training costs remain unobserved.
