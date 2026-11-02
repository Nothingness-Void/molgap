# RML Interfaces

This is the operator reference for `python -m molgap.research_memory`: every
subcommand, the record formats it reads, and how to use them to plan. The design
rules and the authority of RML live in the [RML README](../../research_memory/README.md);
lifecycle internals (`plan`, `finalize`, traces, policies) live in
[LIFECYCLE.md](../../research_memory/LIFECYCLE.md). Source owners are under
`src/molgap/research_memory/` (`cli.py`, `cards.py`, `saved_analysis.py`,
`schemas.py`, `validate.py`).

Samples below were produced on 2026-11-01 from `molgap-desktop` at `6087301a`
(185 trajectories, 173 evidence records). Samples for `levers`, `--compact` and
`experiment_cli catalog --compact` were produced the same day at `5f829540`
(185 cards). Counts will change; formats will not.

## Invocation

```text
.venv\Scripts\python.exe -m molgap.research_memory [--repo-root ROOT] <subcommand> ...
```

- Set `PYTHONIOENCODING=utf-8`. Cards contain non-ASCII text.
- `--repo-root` defaults to the checkout that holds the imported `molgap` package.
  With the shared `.venv`, that is the **primary checkout**, not your worktree.
  In a worktree, pass `--repo-root <worktree>` or set `PYTHONPATH=<worktree>/src`.
  Otherwise you read, and `rebuild` writes, the primary checkout.
- Exit codes: `0` success; `1` a check failed (a `SystemExit` message, or a Python
  traceback from a validation `ValueError`); `2` an argparse usage error.

## Subcommands

Each section lists the exact flags, what the command prints, its exit codes, and
when to use it. Unless noted, the command only reads files.

### Planning queries (read-only, about 0.5 s)

`cards`, `lookup`, `levers` and `gaps` read `research_memory/derived/trajectory_index.json`,
each trajectory record, the trajectory's own bound `paired_analysis.json`, and any
`attribution_label.json` beside its decision; `lookup` and `levers` also read one
historical index (see "Uncarded historical trajectories"). They do not verify hashes. Run
`check --frozen` first if the index might be stale.

#### `cards [--outcome OUTCOME] [--term TERM] [--compact]`

Prints one JSON object per line, one line per trajectory (all outcomes, not only
closed ones). `--outcome` is an exact outcome string. `--term` is one whole word,
case-insensitive, matched anywhere in the card's JSON. No match prints nothing
and exits 0.

| Field | Source |
|---|---|
| `trajectory_id`, `family_id`, `outcome`, `decision_ref` | trajectory index row |
| `mechanism`, `deficiency`, `falsifier` | `hypothesis.changed_mechanism`, `observed_deficiency`, `cheapest_falsifier`; free text or a slug, `null` if absent |
| `related_families` | `hypothesis.related_closed_family_ids` |
| `keywords` | tokens of `mechanism` and `family_id` (see [search terms](#how-search_terms-are-derived)) |
| `reopen_conditions` | `decision.reopen_conditions` |
| `effect_meV`, `ci_meV`, `effect_source` | see [limits](#planning-with-the-evidence) |
| `failure_mode`, `failure_mode_source` | `decision.facts.failure_mode` (`closure_facts`), else an attribution label (`agent_interpretation`), else `null` |
| `search_terms` | at most 8: closure-fact terms, then `failure_mode`, then mechanism-slug and family-id tokens (see [search terms](#how-search_terms-are-derived)) |

```text
{"trajectory_id": "T-pcqm-gptrans-pairnorm-lr-500k-s42-pairnorm_lr5e4", "family_id": "pcqm-gptrans-bondlocal-pairnorm-lr-500k",
 "outcome": "NEGATIVE_UNDER_CONTRACT", "mechanism": "peak-learning-rate", ..., "effect_meV": 0.515, "ci_meV": [0.056, 1.034],
 "effect_source": "experiments/.../pairnorm_lr5e4/paired_analysis.json", "failure_mode": "early_gain_fades",
 "failure_mode_source": "agent_interpretation", "search_terms": ["early_gain_fades", "peak", ...]}
```

Use it to scan prior work by outcome or by a word, for example
`cards --outcome NEGATIVE_UNDER_CONTRACT` or `cards --term readout`.

`--compact` prints one line per card instead of JSON:
`trajectory_id | outcome | family_id | mechanism slug | effect_meV | failure_mode`.
The mechanism slug is the mechanism itself when it is already a slug
(`[a-z0-9_-]+`), else the first 6 lowercase alphanumeric words joined by `-`, else
`-`. The effect is signed with 2 decimals (`+0.52`) or `null`. A missing failure
mode prints `-`. Sample, `cards --outcome NEGATIVE_UNDER_CONTRACT --compact`
(first 3 lines):

```text
T-pcqm-gptrans-pairnorm-lr-500k-s42-pairnorm_lr5e4 | NEGATIVE_UNDER_CONTRACT | pcqm-gptrans-bondlocal-pairnorm-lr-500k | peak-learning-rate | +0.52 | early_gain_fades
T-pcqm-k1-batch-renorm-100k-s42-brn_affine | NEGATIVE_UNDER_CONTRACT | pcqm-k1-batch-renorm-100k | detached-training-mean-correction | +1.65 | endpoint_regression
T-pcqm-k1-extremal-readout-100k-s42-candidate | NEGATIVE_UNDER_CONTRACT | pcqm-k1-extremal-readout-100k | zero-init-max-min-std-branch-added-to-pooled-embedding | +3.33 | -
```

#### `lookup [--mechanism M...] [--failure-mode F] [--family F] [--term T...] [--compact]`

All filters are optional and combine with AND. `--mechanism` takes one or more
words; each is a plain case-insensitive substring test (not whole-word) against one
string: the card's `mechanism`, `family_id` and `failure_mode`, with `null` fields
dropped, joined by single spaces. Every word must match (`--mechanism init noise`).
So `--mechanism lr` also matches `pcqm-gptrans-bondlocal-pairnorm-lr-500k`
through the family id, and `--mechanism readout` matches a family id even when the
mechanism text is a sentence without that word. Free-text mechanisms are matched as
written, so synonyms are missed; try other words, or use `--term` for whole words.
`--failure-mode` is an exact match on the card's `failure_mode`. `--family` is a
case-insensitive substring of `family_id`. `--term` takes one or more words, each a
whole word as in `cards`, and every word must occur in the card. An
unknown failure mode simply matches nothing. Exit 0.

Output is one JSON object:

- `query`: the four filters as given.
- `matches`: full cards.
- `unfalsified`: one entry per matched family that has **no**
  `NEGATIVE_UNDER_CONTRACT` card with a measured effect, with `family_id`,
  `reason`, the first card's `falsifier` and `reopen_conditions`. Read it as "not
  falsified by a measured effect". It does not mean promising.
- `search_terms`: the matches' card `search_terms`, then the 12 most frequent
  keywords, without duplicates (see [search terms](#how-search_terms-are-derived)).

```text
lookup --mechanism readout
  matches: T-pcqm-k1-extremal-readout-100k-s42-baseline, ...-candidate,
           TB-gptrans-500k-frozen-readout-attribution, TB-k1-slot-readout-diagnostic-20261002
  unfalsified[0]: {"family_id": "gptrans-500k-readout-attribution",
                   "reason": "no NEGATIVE_UNDER_CONTRACT with a measured effect", ...}
  search_terms: ["zero", "init", "max", "min", "std", "branch", "added", "pooled", "gptrans", "readout", ...]
```

`--compact` prints the matches as `cards --compact` lines, then one footer line
`# unfalsified families: N; search_terms: <space-separated terms>`. It omits the
`unfalsified` details; drop `--compact` to see them. Sample,
`lookup --mechanism readout --compact` (mechanism slugs shortened here with `...`):

```text
T-pcqm-k1-extremal-readout-100k-s42-baseline | CLOSED | pcqm-k1-extremal-readout-100k | zero-init-max-min-std-branch-...-embedding | null | -
T-pcqm-k1-extremal-readout-100k-s42-candidate | NEGATIVE_UNDER_CONTRACT | pcqm-k1-extremal-readout-100k | zero-init-max-min-std-branch-...-embedding | +3.33 | -
TB-gptrans-500k-frozen-readout-attribution | INCONCLUSIVE | gptrans-500k-readout-attribution | no-encoder-change-compare-frozen-feature | null | -
TB-k1-slot-readout-diagnostic-20261002 | NO_TRAIN | k1-slot-readout-diagnostic | no-change-observational-hooks-and-exact | null | discriminator_missing
# unfalsified families: 2; search_terms: zero init max min std branch added pooled gptrans readout attribution discriminator_missing slot diagnostic embedding extremal
```

#### How `search_terms` are derived

All terms come from fields already in the records (`cards.py`). Nothing is read
from decision prose.

1. **Keywords** (`keywords` on each card): `mechanism` and `family_id` are joined,
   lowercased and split on every character outside `a-z0-9`. A token is kept when
   it has more than 2 characters, is not all digits, is not a stop word (`the`,
   `and`, `for`, `with`, `training`, `model`, `fixed`, `selected`, ... in `_STOP`)
   and does not match `s<digits>`, `<digits>k`, `pcqm` or `r<digit>`. The list is unique and
   sorted alphabetically.
2. **Card `search_terms`**, in this order, without duplicates, cut to 8:
   closure-fact `decision.facts.search_terms`; the card's `failure_mode` (from
   closure facts or an attribution label); tokens of the mechanism, used only when
   the whole mechanism is a slug (`[a-z0-9_-]+`; a sentence contributes nothing
   here); tokens of `family_id`. Slug and family tokens keep their order of
   appearance and pass the same filter as keywords.
3. **`lookup` `search_terms`**: every match's card `search_terms` in match order,
   then the 12 most frequent keywords across the matches (ties keep first-seen
   order), without duplicates. There is no length cap.

On the sample date no file under `experiments/` held a non-empty
`search_terms` list, so every term came from `failure_mode` and id tokens.

#### `levers --family PATTERN`

A read-only table of tested levers for every family whose id matches `PATTERN`.
`--family` is required. Without `*` it is a case-insensitive substring of
`family_id`; with `*` it is a case-insensitive glob over the whole id
(`fnmatch`, so `pcqm-k1-*-100k` must match the full id). Exit 0, also when nothing
matches (only the header prints).

- Columns: `family | arm | outcome | effect_meV | failure_mode | mechanism`.
- Families are sorted by id. `arm` is the trajectory id with the `T-`/`TB-`/`TC-`
  prefix and the family id removed; when the id does not start with the family id,
  the stripped id stays (see the pairnorm sample).
- Same-run reference arms (`CLOSED`) are left out unless the family has nothing
  else.
- `effect_meV` is the card effect. A `NEGATIVE_UNDER_CONTRACT` row without a
  recorded effect prints `null (see <decision_ref>)`; open that decision file for
  the reason. Other missing effects print `null`. `failure_mode` prints `-` when
  absent. `mechanism` is the compact slug.

`levers --family 'pcqm-k1-*-100k'` (first 9 of 16 lines):

```text
family | arm | outcome | effect_meV | failure_mode | mechanism
pcqm-k1-batch-renorm-100k | s42-brn_affine | NEGATIVE_UNDER_CONTRACT | +1.65 | endpoint_regression | detached-training-mean-correction
pcqm-k1-extremal-readout-100k | s42-candidate | NEGATIVE_UNDER_CONTRACT | +3.33 | - | zero-init-max-min-std-branch-added-to-pooled-embedding
pcqm-k1-hop-shell-100k | s42-hop_ordered | NEGATIVE_UNDER_CONTRACT | -1.12 | - | exact-shortest-hop-content-partition
pcqm-k1-init-noise-100k | s43-init_s43 | NEGATIVE_UNDER_CONTRACT | -1.63 | - | initialization-seed-noise
pcqm-k1-initial-residual-100k | s42-initial_residual | NEGATIVE_UNDER_CONTRACT | +0.71 | - | gated-initial-residual-h0-into-conv-input-blocks-2-to-9
pcqm-k1-lr2e4-100k | s42-lr2e4_i43 | NEGATIVE_UNDER_CONTRACT | null (see experiments/pcqm_k1_lr2e4_100k/terminal_decision.md) | - | lower-peak-lr
pcqm-k1-lr8e4-warmup-100k | s42-lr8e4w3_i43 | NEGATIVE_UNDER_CONTRACT | null (see experiments/pcqm_k1_lr8e4_warmup_100k/terminal_decision.md) | - | higher-peak-lr-with-warmup
pcqm-k1-message-droppath-100k | s42-message_droppath | NEGATIVE_UNDER_CONTRACT | +2.37 | endpoint_regression | graph-wise-neighbor-aggregate-droppath
```

`levers --family pairnorm` (substring):

```text
family | arm | outcome | effect_meV | failure_mode | mechanism
pcqm-gptrans-bondlocal-pairnorm-100k | s42-pair_update_norm | POSITIVE_UNDER_CONTRACT | -3.14 | boundary_pass | pair-update-channel-normalization
pcqm-gptrans-bondlocal-pairnorm-lr-500k | pcqm-gptrans-pairnorm-lr-500k-s42-pairnorm_lr5e4 | NEGATIVE_UNDER_CONTRACT | +0.52 | early_gain_fades | peak-learning-rate
pcqm-k1-gptrans-pairnorm-fusion-500k | k1-gptrans-pairnorm-fusion-500k-20261027 | NO_TRAIN | null | comparator_missing | fixed-averaging-of-saved-model-predictions
pcqm-pairnorm-prediction-router-500k | 20261027 | NO_TRAIN | null | below_gate | existing-hard-and-convex-soft-cross
```

The init-noise rows show why effect needs reading: their lever is seed noise, so
`-1.63` is a seed difference, not a method gain.

### Addon catalog (`experiment_cli`)

`.venv\Scripts\python.exe -m molgap.experiment_cli catalog [--compact]` is owned by
`src/molgap/experiment_cli.py` ([CLI reference](EXPERIMENT_CLI.md)). It reads code
only and exits 0. Use it before planning to see which family or addon the
scientific input compiler can run without new code.

Plain output is JSON `{"families": [...], "addons": [...], "submitted": false}`.
Besides the declared contract fields, each entry has:

- `runnable_via_compiler`: `true` when the compiler accepts it. A family is runnable
  when it is a shared recipe owner (`family_training_runner.INTERFACES`). An addon
  is runnable when a runnable family of the same name registers that exact
  name/version.
- `runnable_reason`: for families `shared recipe owner`, `no training adapter`, or
  `not a shared recipe owner (<module>)`. For addons
  `registered for compiler family <name>/<version>`,
  `empty execution_modes (declaration only)`, `no compiler family for <family>`,
  or `execution_modes cover versions <v,...>; compiler family is <name>/<version>`.
- Families also carry `execution_support`: `DECLARATION_ONLY`,
  `CPU_SYNTHETIC_ONLY` or `REGISTERED_REQUIRES_QUALIFICATION`.

`--compact` prints `{"lines": [...]}`, one line per family and then per addon:
`family <name>/<version> runnable=<bool> | <reason> | <purpose>` or
`addon <name>/<version> (<family>) runnable=<bool> | <reason> | <purpose>`. The
purpose is the first docstring line of the entry's `source_module`. On the sample
date it printed 26 lines, 9 of them `runnable=false`. Sample (excerpt):

```text
family neural_atom_k1/3 runnable=true | shared recipe owner | Registered shared K1/GPTrans entrypoints; platform skills own job operations.
family gptrans_t/2 runnable=true | shared recipe owner | Registered shared K1/GPTrans entrypoints; platform skills own job operations.
family edge_state_gps/1 runnable=false | no training adapter | OGB EdgeState architecture variants; no training or submission authority.
family gptrans_t/1 runnable=false | not a shared recipe owner (molgap.gptrans_screen_workflow) | GPTrans-T core adapted to MolGap's accepted PyG graph batches.
addon k1_node_scale/1 (neural_atom_k1) runnable=true | registered for compiler family neural_atom_k1/3 | Small registered slot example shared by the two accelerated backbones.
addon k1_pair_value/1 (neural_atom_k1) runnable=false | empty execution_modes (declaration only) | Value-decoupled pair token on the existing K1; provenance in K1_ADAPTER.md.
```

`runnable=false` does not forbid the work. It means the change needs its existing
typed builder or a new addon module and catalog entry (AGENTS infrastructure
budget rule).

#### Uncarded historical trajectories

Older K1 trajectories, mostly from the server branch, have no desktop trajectory
record, so they have no card and `lookup`/`levers` cannot list them. Their one
index is the table under "Indexed trajectory and attempt coverage" in
`experiments/pcqm_k1_complete_local_audit/historical_evidence.md` (87 rows, 27 of
them carded on 2026-11-01). Example: `TC-k1-degree-balance-100k-s42`,
`POSITIVE_BELOW_GATE`, +1.28 meV against the 3 meV gate on K1 v4 100K, and a
500K regression (same file, lines 151-153).

When a query matches uncarded rows of that table, `lookup` (plain and
`--compact`) and `levers` add one line, and plain `lookup` JSON adds the same text as
`historical_pointer`:
`# not in cards: N historical trajectories match in <file>: <id> <outcome>, ... (first 5)`.
`--mechanism` is a substring and `--term` a whole word of the row text. `--family`
matches `pcqm-` + the row id without track prefix and `-sNN`/`-vN`/`-rN`
suffixes, so `pcqm-k1-*-100k` lists 37 rows. `--failure-mode` alone prints no pointer.
Open the file for effects; these rows carry no effect field.

```text
lookup --mechanism degree --compact   (last line)
# not in cards: 2 historical trajectories match in experiments/pcqm_k1_complete_local_audit/historical_evidence.md: TB-k1-ssma-qualification-s42-v1 NO_TRAIN, TC-k1-degree-balance-100k-s42 POSITIVE_BELOW_GATE
```

#### `gaps [--compact]`

Exit 0. Prints `{"sparse": [...], "contradictions": [...]}`. `--compact` prints
one line per item instead: `sparse | <family_id> | n=<N> with_effect=<N>` and
`contradiction | <arm> | <id> <scale>k <outcome> <effect> vs <id> <scale>k <outcome> <effect>`.

- `sparse`: families with exactly one trajectory, as
  `{family_id, n_trajectories, n_with_effect}` (58 on the sample date).
- `contradictions`: the same named arm at two scales with different
  `POSITIVE_UNDER_CONTRACT`/`NEGATIVE_UNDER_CONTRACT` outcomes. Arms are matched by
  trajectory id with the track prefix, `-NNNk` and `-sNN`/`-seedNN` removed. Each
  entry is `{key, a, b, kind: "outcome_differs_across_scale_needs_review"}`, for
  example `gptrans-noisy-nodes` (100K positive, 500K negative).

Both lists prompt review. They are not findings.

### Saved-prediction analysis

#### `analyze --op {paired,ensemble-curve,residual-corr,summary,floor} --predictions P... --role-id ROLE --out OUT [--pair CAND=REF]...`

This command does no training and writes nothing to RML. Each `--predictions`
file is a torch file with `source_idx`, `target_eV` and `prediction_eV`. Every file
must have the same rows and targets. You need at least two files. Pass bare paths,
or `ARM=PATH` to name the arms. `--pair CAND=REF` can repeat and is used only by `summary`.

The command refuses a missing `--role-id` or one of `official_validation`,
`test_dev` and `test_challenge` (exit 1). It also refuses an `--out` under
`research_memory/` or `experiments/` (exit 1). Write to scratch. stdout is
`{"op", "rows", "out", "posthoc": true}`. The `--out` file holds `op`, `role_id`,
`inputs` (path and SHA-256), `row_count`, `source_idx_sha256`, `results` and
`posthoc: true`. The result is never a gate decision.

| `--op` | `results` |
|---|---|
| `paired` | first file is the reference; per candidate `mae_meV`, `delta`, `ci95`, `probability_better`, `n_bootstrap`, `seed` (delta < 0: candidate better; row bootstrap only) |
| `ensemble-curve` | `singles_meV`, `mean_single_mae_meV`, `mae_by_n_members_meV`, `gain_all_members_meV` |
| `residual-corr` | `members`, `pair_correlation`, `mean_pair_correlation` |
| `summary` | `singles_meV`, `n_inits`, `D_meV` (max-min single), `ensemble_mae_meV`, `ensemble_gain_meV`, `fit` (`a + b/N`, needs 3 or more members), `pairs` (from `--pair`), `m_meV` |
| `floor` | `members_used` (bit-identical reruns counted once), `shared_fraction_of_mse`, `all_wrong_rows`, `oracle_mae_meV` |

Sample (K1 100K init noise, `ref_s42` vs `init_s43`, 50 000 rows): `summary` gave
`D_meV 1.63`, pair delta `-1.63` with CI `[-2.56, -0.71]`, `ensemble_gain_meV 5.78`;
`floor` gave `shared_fraction_of_mse 0.924`.

### Integrity and freshness

#### `validate`

No flags. This is the full catalog validation: schemas, unique ids, pointers and
comparison-artifact bytes. It takes about 20 s and needs every bound local
artifact. Prints record counts and `known_gaps`:

```text
{"evidence": 173, "trajectories": 185, "costs": 410, "roles": 609, "traces": 86, "ready": 0,
 "comparison_prelaunch": 0, "comparison_readiness": 29, "reference_bundles": 28, "target_transform_assets": 22,
 "known_gaps": ["experiments/pcqm_k1_init_noise_100k/terminal_decision.md"]}
```

Exit 0 on success. On failure it exits 1 with a traceback naming the first bad
record. In a fresh worktree it fails with `required repository pointer is missing:
.../best_model.pt`, because ignored weights live only in the primary checkout. Run
it there, or with `--repo-root` pointing at it.

#### `check --frozen [--portable]`

`--frozen` is required. The check recompiles the derived indexes in memory and
compares their bytes with `research_memory/derived/`. It does not verify artifact
bytes, so it passes in a fresh worktree. It takes about 30 s. Success prints
`RML derived outputs are frozen and current` and exits 0. Otherwise it exits 1
with `stale derived outputs: <files>`.

`--portable` also checks that every RML input, the runtime and the derived files
are in Git HEAD unchanged, and that every artifact V5 evidence claims is local. It
takes about 50 s. On the sample date it exited 1 in the primary checkout, listing
103 `V5 evidence claims missing local artifacts` lines for evicted checkpoints
(see [storage lifecycle](LOCAL_STORAGE_RETIREMENT.md)). Report that honestly. Do
not hydrate artifacts to make it pass.

Run `check --frozen` before a commit or handoff that touches canonical records, and
before you trust `cards`/`lookup` output.

#### `rebuild`

No flags. The command regenerates all 12 files in `research_memory/derived/`
atomically and prints `{"status": "rebuilt", "files": [...]}`. It takes about
30 s and does not verify artifact bytes. Run it once after a standalone canonical
edit or import, never per epoch. Shared terminal closure already rebuilds, so do
not rebuild again after a successful closure.

#### `terminal-rebuild --trajectory PATH [--trajectory PATH ...]`

PATH is a finalized `trajectory.json` or its `rml_finalized/` directory. A bare
trajectory id is rejected. The command re-verifies the immutable finalization
receipt and refreshes only the derived outputs. It prints one result per
trajectory (`pipeline_status`, `finalization_status: ALREADY_FINALIZED`,
`validation_status`, `outcome`, `replay_pool_delta`, `next_decision`). It exits 1
unless every result is `COMPLETE`. Use it only to recover a closure whose index
update failed.

#### `status`

No flags. Prints `research_memory/derived/research_summary.md`: evidence counts,
trajectories by outcome, READY transfer, roles and cost totals. Exits 1 with
`research summary is missing; run rebuild` if the file is absent.

#### `doctor [--strict]`

Prints `completeness_report.json` with these keys: `broken_required_pointers`,
`cost_completeness`, `counts_by_severity`, `format`, `issues[{code, severity,
count, items}]` and `ready_prerequisites_missing`. It exits 1 on any `ERROR`
issue, and with `--strict` also on any `WARN` issue. On the sample date the plain
command exited 0 and `--strict` exited 1. The issues were
`historical_partial_record` (20), `incomplete_native_cost_measurement` (142),
`missing_cost_event` (11), `missing_explicit_role_identity` (47) and
`trajectory_without_v5_evidence` (12), all WARN, plus
`trace_unavailable_for_backtest` (99, INFO).

### Replay and backtests (read-only, run `validate` first internally)

These commands call full validation, so they need the local artifacts that
`validate` needs (about 15-25 s each).

- `replay-pool`: prints the strict comparable replay pool (`comparable_group_count`,
  `entries[{axis, capability, comparability_identity, ...}]`, `action_entries`). The
  output is large (about 73 000 lines).
- `backtest-screening`: prints the screening backtest (`excluded_traces[{reasons}]`,
  `confounding_warnings`, `candidate_prefix_rule`, ...).
- `backtest-policy [--policy-id ID [--policy-version V]]`: backtests registered
  policies against the pool. It prints `format: molgap-rml-policy-backtest-v1`,
  `policy_activated: false` and `reports[]` with false-stop/false-promote counts
  and device-hours. `--policy-version` without `--policy-id`, or an unknown pair,
  raises an error (exit 1).

### Writing commands

These write canonical or handoff records. Details are in
[LIFECYCLE.md](../../research_memory/LIFECYCLE.md); the normal closure path is
`platforms/local/accept_run.py` ([CLI reference](EXPERIMENT_CLI.md)).

| Command | Flags | Writes |
|---|---|---|
| `plan` | `--spec SPEC --output OUTPUT` | a prospective trajectory from a planning spec |
| `finalize` | `--trajectory T --terminal TERMINAL [--trace TRACE]` | an immutable `rml_finalized/` set |
| `terminal-pipeline` | `--trajectory T --terminal TERMINAL [--trace TRACE] [--trace-source SRC] [--arm ARM]` | closes one arm and updates derived RML; exit 1 unless `pipeline_status` is `COMPLETE` |
| `recover-trace` | `--source SRC [--source ...] --spec SPEC --output OUTPUT` | a canonical trace recovered from retained logs |
| `package-ready` | `--trajectory ID` | `handoff/READY_FOR_DESKTOP.json` beside the trajectory; prints its path. An unknown id exits 1 (`unknown trajectory: ID`) |

## Data contracts

### Record kinds

The compiler discovers records under `experiments/` by filename:
`v5_evidence.json` (evidence), `trajectory.json`, `trace_manifest.json`,
`comparison_readiness.json`, `comparison_readiness_prelaunch.json`,
`reference_bundle.json`, `target_transform.json`, `costs/*.json`, `roles/*.json`
and `handoff/READY_FOR_DESKTOP.json`. A closed arm keeps them under
`runs/<run>/prospective/<arm>/rml_finalized/`, together with `finalization.json`
(the receipt) and `terminal_input.json`.

- **Trajectory** (`molgap-trajectory-v1`): one question or arm, with `hypothesis`,
  `state_at_start`, `actions`, `result`, `decision` and optional comparison
  metadata. `record_mode` is `prospective` (published before work) or
  `retrospective_partial` (honest import of history).
- **Evidence** (V5 envelope): measured results and bound artifacts with SHA-256.
  `role_use` per protected role.
- **Closure**: `decision.outcome`, `decision.decision_ref` (the human decision
  file), `reopen_conditions`, `next_allowed_actions`, optional `decision.facts`.
  The closure input to `accept_run.py --closure-input` is
  `{"arms": {ARM: {"outcome", "decision_ref", "facts_ref"?}}, "role_audit"?}`.

### Outcome vocabulary

From `TRAJECTORY_OUTCOMES` in `schemas.py`. Counts are from the sample date.

| Outcome | Meaning | n |
|---|---|---|
| `ACTIVE` | open; no terminal decision yet | 1 |
| `NO_TRAIN` | closed without training (feasibility, cache, CPU-only or profiling work) | 57 |
| `POSITIVE_UNDER_CONTRACT` | the candidate passed the contract's gate | 9 |
| `POSITIVE_BELOW_GATE` | favorable but below the material gate | 0 |
| `NEGATIVE_UNDER_CONTRACT` | the candidate failed the contract's gate; a measured negative | 52 |
| `INCONCLUSIVE` | evidence could not decide the question | 23 |
| `STOP_FOR_COST` | stopped by a frozen cost or overhead gate | 0 |
| `DUPLICATE_EVIDENCE` | the question was already answered by existing evidence | 0 |
| `INFRASTRUCTURE_ONLY` | the run produced no science (runtime or infra failure, or speed work) | 10 |
| `READY_FOR_DESKTOP` | fail-closed transfer package to desktop | 0 |
| `CLOSED` | arm closed without a candidate verdict, typically the same-job **control/reference** arm | 33 |

`CLOSED` matters for effects: an effect is taken only against a reference arm that
closed `CLOSED`.

### `decision.facts` (`molgap-closure-facts-v1`)

These are optional per-arm closure facts, validated by `validate_closure_facts`.
Allowed keys: `format`, `failure_mode`, `search_terms`, `effect_meV`, `ci_meV`
and `source`.

| Field | Rule |
|---|---|
| `format` | `"molgap-closure-facts-v1"` |
| `failure_mode` | one enum value below, or `null` |
| `search_terms` | `null` or at most 8 non-empty strings |
| `effect_meV` | `null` or a finite number |
| `ci_meV` | `null` or `[low, high]` finite numbers |
| `source` | `null` or a repository path to the measured file |

At closure (`experiment_terminal_staging._closure_facts`), a non-null effect must
equal a `pairs[]` entry (`delta`, `ci95`) in `source`. That entry's candidate must
be this arm, its reference must close `CLOSED`, and both prediction hashes must be
among `source`'s `inputs`. Anything else is rejected, and effect, CI and source
must be `null`. `experiment_cli draft-closure` writes
`closure_facts_<arm>.json` with the measured numbers copied and `failure_mode` and
`search_terms` left `null`. The closure author fills those in.

`failure_mode` enum (`FAILURE_MODES`):

| Value | Meaning |
|---|---|
| `infra_failure` | runtime or infrastructure failure; no usable predictions |
| `gate_or_cost_stop` | a frozen gate (overhead, cost) stopped the run before the science |
| `lost_at_scale` | a gain at smaller scale did not hold at larger scale |
| `early_gain_fades` | the candidate leads early in training and loses by the selected endpoint |
| `endpoint_regression` | same-job candidate worse at the endpoint; the paired interval excludes zero |
| `below_gate` | favorable direction but below the material gate |
| `confounded_change` | more than one factor changed, so the cause is not isolated |
| `comparator_missing` | no matched comparator exists for the claim |
| `discriminator_missing` | candidate causes listed but the discriminating run was not done |
| `boundary_pass` | passes the gate only near its threshold |

The code defines only the enum. The meanings above come from how the 43 existing
labels use each value.

### `attribution_label.json` side files

These are weak-evidence labels beside a decision file (in the decision's
directory). There are 43 on the sample date. `cards` uses a label only when the
trajectory has no closure-fact `failure_mode` and the trajectory id is in
`trajectory_ids`.

| Field | Meaning |
|---|---|
| `trajectory_ids` | trajectories the label covers |
| `failure_mode` | enum value, or `null` = no model failure to label (profiling, infra-only, speed or NO_TRAIN work); 21 of 43 are `null` |
| `secondary_modes` | further enum values; not shown on cards |
| `source` | `"agent_interpretation"`: an agent read the attribution, nothing measured it |
| `basis` | `{path, sha256}` of the attribution text the label was read from |
| `labeled_at`, `note` | date and one-line reason |

No code checks a label's schema or `basis.sha256`. On the sample date all 43
basis hashes matched their files.

### Errata

An erratum excuses one known narrative hash mismatch. It lives at
`experiments/<q>/errata/*.json`:

```json
{"format": "molgap-erratum-v1", "target_locator": "experiments/<q>/<file>.md",
 "artifact_class": "narrative", "effect": "narrative_only",
 "recorded_sha256": "<digest bound in records>", "observed_sha256": "<digest of current file>",
 "what_is_known": "...", "what_is_not_known": "...", "author": "...", "date": "...", "note": "..."}
```

Limits (`validate._load_errata`):

- The target must be a `.md` file inside the same `experiments/<q>/` directory as
  the erratum, and its locator must use forward slashes. Any other erratum makes
  `validate` fail.
- It applies only when both the recorded digest and the current file's digest
  match exactly. It never covers checkpoints, predictions, JSON records or another
  experiment.
- Only `validate` reads errata. It reports the file under `known_gaps` instead of
  failing. Strict acceptance, `check` and the closure path ignore errata.
- The single erratum today covers
  `experiments/pcqm_k1_init_noise_100k/terminal_decision.md`.

### Hash classes

| Class | Examples | Checked by | Strength |
|---|---|---|---|
| Bound artifact SHA-256 | V5 evidence artifacts, `comparison_readiness` bindings, `trace_artifact_sha256`, reference bundle digests | `validate`, `replay-pool`/backtests, strict acceptance | strict, fail-closed. a NUL-free file in Git `HEAD` also matches its uniform LF or CRLF form, whatever its text attribute; external, NUL-containing and mixed-ending files, and `finalization.json` receipts, must match byte for byte |
| Closure input hashes | `finalization.json` / `terminal_input.json` digests of decision, facts, role audit, manifest | strict recheck (`accept_run.py --strict`) | strict; errata do not apply |
| Erratum digest pair | `recorded_sha256`/`observed_sha256` | `validate` only | exact match required, effect is narrative only |
| Index freshness | derived bytes, `source_digest` | `check --frozen` | strict for the index, says nothing about artifacts |
| Advisory | `attribution_label.basis.sha256`, `analyze` `inputs[].sha256`, derived `effect_source` paths | nothing | provenance for a reader; not verified |

### Derived indexes

`research_memory/derived/` contains generated files that you must never edit:
`trajectory_index.json`, `trajectory_graph.json`, `cost_ledger.json`,
`role_reuse_index.json`, `reference_reuse_index.json`,
`ready_for_desktop_index.json`, `screening_backtest.json`, `replay_pool.json`,
`policy_backtest.json`, `completeness_report.json`, `research_summary.json` and
`research_summary.md`.

- After a standalone canonical edit or import, run `rebuild` once.
- After `accept_run.py` closure, do not rebuild. The closure updates the indexes.
  If its index update failed, run `terminal-rebuild`.
- Before a commit or handoff, run `check --frozen`. Use `--portable` when the
  receiver needs every input committed and every claimed artifact present.
- Before trusting `cards`/`lookup`/`gaps`, run `check --frozen` if canonical
  records changed since the last rebuild.

## Planning with the evidence

1. `cards --outcome NEGATIVE_UNDER_CONTRACT` and `cards --term <word>`: scan what
   was tried.
2. `lookup --mechanism <word>` and `lookup --failure-mode <mode>`: collect related
   work, why it failed (`failure_mode_source`), `unfalsified` families and
   `search_terms`.
3. `gaps`: check whether your idea falls in a single-trajectory family or a
   cross-scale contradiction.
4. Open each match's `decision_ref`, then its `effect_source` and the
   `attribution.md` named in the label's `basis`. Cards are pointers. The
   decision file is the evidence.
5. Search papers with the `search_terms` plus your own mechanism words. If you
   delegate paper reading, split it across parallel subagents.
6. Publish a prospective trajectory before any work
   ([Research Protocol](RESEARCH_PROTOCOL.md#new-research-question-protocol)).

Honest limits:

- **Effect** is non-null only on 15 of 185 cards.
  It comes from closure facts or from the trajectory's own `paired_analysis.json`,
  and only when the reference arm closed `CLOSED`. Two lever arms that differ by
  seed measure seed noise, so their effect stays `null`. A `null` effect is
  missing data. It is not zero.
- **search_terms** are built from fields that already exist (closure facts,
  failure mode, mechanism slug, family id). They add no new knowledge and are only
  as good as those fields.
- **mechanism** is partly free text (some cards hold a sentence, some a slug).
  Substring matching misses synonyms, so try several words.
- **Labels** are `agent_interpretation`, which is weak evidence. 24 of 185 cards
  carry one, and none currently comes from closure facts. Confirm against the
  decision file before relying on it.
- `unfalsified` and `gaps` only say where evidence is thin. They do not show that
  an idea is promising.

## When verifiers disagree

| Situation | Which one wins |
|---|---|
| `validate` passes with a `known_gaps` entry, strict recheck `BLOCKED` | **Strict recheck** for replay, comparison and promotion. `validate`'s pass only says the catalog is consistent apart from a narrative file |
| `check --frozen` passes, `validate` fails | `validate`. Freshness is not artifact integrity |
| `validate` passes, `check --frozen --portable` fails | both are correct; the run is valid here but not portable. Report the gap |
| card or label vs decision file | the decision file ([AGENTS precedence](../../AGENTS.md#authority)) |

Example on 2026-11-01:
`accept_run.py --strict --execute --profile D:/molgap-machine-profile.json
--run-id pcqm-k1-init-noise-100k-s43 --attempt-id v1` exits 1 with
`replay_readiness: BLOCKED`, because both arms report `bound artifact SHA mismatch:
experiments/pcqm_k1_init_noise_100k/terminal_decision.md`. Native qualification
is still `VERIFIED` for both arms. `validate` on the same tree exits 0 and lists
that file in `known_gaps`. This behavior is intended, and the run stays blocked
for replay.
