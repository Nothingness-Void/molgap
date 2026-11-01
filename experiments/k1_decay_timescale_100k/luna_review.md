# K1 decay front-door verification — 2026-11-02

**Result: do not continue unchanged.** Checkout `C:/Users/Adminn/Documents/molgap` is on `molgap-server` at `9550a2bcd03555f873daaca845ba7ec747cceec1`; `5ee57ea86` is an ancestor. The worktree already had unrelated untracked generated artifacts; they were left untouched.

The authored experiment currently gives `shared.preset_ref` as a `{path, sha256}` object, which matches the resolver's pinned-reference shape. A bare string fails that check (`src/molgap/experiment_presets.py:194-207`). The named preset points to `experiments/pcqm_structural_bias_norm_100k/k1/training_recipe_post_bn.json`.

The single read-only `read_recipe` call returned `ValueError: Missing or unknown shared training recipe fields`. The parser strips its declared metadata, but does not strip `resume_chunks`; that top-level key is not a `FamilyRecipe` field and is rejected by `FamilyRecipe.from_dict` (`src/molgap/family_training_runner.py:32-64,134`; `src/molgap/family_training_recipe.py:361-371`; recipe line 35). This confirms the second reported blocker occurs before freeze.

Routine preparation needs compatible recipe bytes and matching new pins (or an explicitly approved parser compatibility change); do not rewrite the frozen preset in place. The specified `E:/MolGap/data` path is also absent, while `E:/MolGap/machine-profile.json` exists. No alternate roots were inspected. No submission was retried; no tests, training, inference, network, credentials, or infrastructure/configuration changes were used.
## Concurrent worktree change

After the initial clean status for this path, `experiments/k1_decay_timescale_100k/experiment.json` appeared modified. The observed diff changes the reference arm from `wd1e4` to `reference`, updates the comparison question, and changes the arm declarations. I did not make or revert this change and left the file preserved. Its origin is unknown. No submission was attempted with this updated declaration.