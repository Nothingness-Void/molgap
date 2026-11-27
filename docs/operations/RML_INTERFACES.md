# RML Interfaces

Operator reference for `molgap.research_memory`: command semantics, record schemas, planning recipe, and verifier
precedence. Design authority lives in [research_memory/README.md](../../research_memory/README.md); lifecycle internals
live in [research_memory/LIFECYCLE.md](../../research_memory/LIFECYCLE.md). Source owners are under
`src/molgap/research_memory/` (`cli.py`, `cards.py`, `saved_analysis.py`, `schemas.py`, `validate.py`).

## Invocation

```text
.venv\Scripts\python.exe platforms/local/run.py --module molgap.research_memory -- [--repo-root ROOT] <subcommand> ...
```

- Run `<subcommand> --help` for authoritative flags and valid schema choices (`--outcome`, `--failure-mode`, `--op`).
- Set `PYTHONIOENCODING=utf-8` (cards can contain non-ASCII text).
- `--repo-root` defaults to the current directory's git root; always use `platforms/local/run.py` in worktrees so the
  worktree's own code runs.
- Exit codes: `0` success; `1` failed check (`SystemExit` or validation `ValueError`); `2` argparse usage/choice error.

## Subcommands

### Planning and query (read-only, ~0.5 s)

`cards`, `lookup`, `levers`, and `gaps` read `research_memory/derived/trajectory_index.json`, each trajectory, its
bound `paired_analysis.json`, and any side `attribution_label.json`. They do not verify artifact hashes (run
`check --frozen` first if canonical records changed). `--ref REV` (`cards`, `lookup`, `gaps`) merges cards committed
only on git revision `REV` (e.g. `origin/molgap-server`) without validating or copying `REV`; local wins.

| Command | Behavior and output semantics (use `--help` for flags) |
|---|---|
| `cards` | One JSON card per line (or `--compact`: `id \| outcome \| family \| mechanism slug \| effect_meV \| failure_mode`). `--outcome` validates against `TRAJECTORY_OUTCOMES`. |
| `lookup` | AND-combined filters (`--mechanism`, `--term`, `--family`, `--failure-mode` validated against `FAILURE_MODES`). Plain JSON returns `{query, matches, unfalsified, search_terms}` (plus `historical_pointer` when uncarded rows match). |
| `levers` | Table `family \| arm \| outcome \| effect_meV \| failure_mode \| mechanism` for `--family` (substring, or `fnmatch` glob with `*`). Drops `CLOSED` reference arms unless nothing else exists; negative arms without recorded effect print `null (see <decision_ref>)`. |
| `gaps` | Lists `sparse` (single-trajectory families) and `contradictions` (same arm `POSITIVE_UNDER_CONTRACT` vs `NEGATIVE_UNDER_CONTRACT` across scales). Prompts review; not findings. |

- **`search_terms` derivation (`cards.py`):** built strictly from structured fields (never decision prose): closure-fact
  `search_terms`, then `failure_mode`, then slug/family tokens (>2 chars, non-digit, non-stopword, excluding `s<n>` /
  `<n>k` / `pcqm` / `r<n>`), capped at 8 per card; `lookup` appends the 12 most frequent match keywords.
- **Uncarded historical trajectories:** older K1/server runs without desktop trajectory records are indexed in
  `experiments/pcqm_k1_complete_local_audit/historical_evidence.md`. When `lookup` or `levers` matches uncarded rows,
  it appends `# not in cards: N historical trajectories match in <file>: ...` (open that file for effects).
- **Addon catalog:** run `platforms/local/run.py -- catalog --compact` ([CLI reference](EXPERIMENT_CLI.md)) before
  planning to see which families/addons the compiler runs without new code.

### Saved-prediction analysis (`analyze`)

`analyze` runs post-hoc calculations over >=2 torch files with matching `source_idx`, `target_eV`, and `prediction_eV` (bare paths or `ARM=PATH`); no training or RML writes.
Refuses missing `--role-id` or protected roles (`official_validation`, `test_dev`, `test_challenge`) and refuses `--out` under `research_memory/` or `experiments/` (exit `1`). Output carries `posthoc: true` and is never a gate decision.

| `--op` | `results` summary |
|---|---|
| `paired` | First file is reference; per candidate `mae_meV`, `delta` (<0 better), `ci95`, `probability_better`, `n_bootstrap`, `seed` (row bootstrap only) |
| `ensemble-curve` | `singles_meV`, `mean_single_mae_meV`, `mae_by_n_members_meV`, `gain_all_members_meV` |
| `residual-corr` | `members`, `pair_correlation`, `mean_pair_correlation` |
| `summary` | `singles_meV`, `n_inits`, `D_meV`, `ensemble_mae_meV`, `ensemble_gain_meV`, `fit` (`a + b/N`, >=3 members), `pairs` (from `--pair`), `m_meV` |
| `floor` | `members_used` (bit-identical reruns counted once), `shared_fraction_of_mse`, `all_wrong_rows`, `oracle_mae_meV` |

### Integrity, maintenance, replay and writing

| Command | Mode | Key behavior and gotchas |
|---|---|---|
| `check --frozen [--portable]` | Read-only (~30 s) | Recompiles derived indexes in memory and checks byte equality against `research_memory/derived/`; works in fresh worktrees. `--portable` also checks Git HEAD cleanliness and local artifact presence (reports evicted checkpoints honestly; never hydrate to force pass). |
| `validate` | Read-only (~20 s) | Full schema, ID, pointer, and bound-artifact byte check; requires primary checkout artifacts (fails in worktrees missing ignored weights). |
| `status` | Read-only | Prints `research_memory/derived/research_summary.md` (exits `1` if missing). |
| `doctor` | Read-only / clean | Prints `completeness_report.json` (exits `1` on `ERROR`, or also `WARN` with `--strict`); `--clean-staging [--apply]` lists/cleans unclosed staging. |
| `rebuild` | Write (~30 s) | Atomically regenerates `research_memory/derived/`. Run once after standalone canonical edits/imports, **never** after normal closure (closure already rebuilds). |
| `terminal-rebuild` | Recovery write | Pass finalized `trajectory.json` or `rml_finalized/` path (not a bare ID). Re-verifies finalization receipt and refreshes derived outputs after a failed closure index update. |
| `replay-pool`, `backtest-screening`, `backtest-policy` | Read-only (~20 s) | Run `validate` internally first, so they need primary checkout artifacts. `replay-pool` stdout is very large (~73k lines); `backtest-policy` prints `policy_activated: false`. |
| `plan`, `finalize`, `terminal-pipeline`, `legacy-terminal`, `recover-trace`, `package-ready` | Low-level writes | Called by [LIFECYCLE.md](../../research_memory/LIFECYCLE.md) / `platforms/local/accept_run.py` ([CLI reference](EXPERIMENT_CLI.md)). Use `--help` for flags. |

## Data contracts

### Record kinds

Discovery scans `experiments/` for `v5_evidence.json`, `trajectory.json`, `trace_manifest.json`,
`comparison_readiness.json`, `comparison_readiness_prelaunch.json`, `reference_bundle.json`, `target_transform.json`,
`costs/*.json`, `roles/*.json`, and `handoff/READY_FOR_DESKTOP.json`. Closed arms keep finalized copies under
`runs/<run>/prospective/<arm>/rml_finalized/` with `finalization.json` and `terminal_input.json`.

### Outcome vocabulary

Defined in `TRAJECTORY_OUTCOMES` (`schemas.py`; enforced by CLI `--outcome`):

| Outcome | Meaning |
|---|---|
| `ACTIVE` | Open; no terminal decision yet |
| `NO_TRAIN` | Closed without training (feasibility, cache, CPU-only, or profiling) |
| `POSITIVE_UNDER_CONTRACT` | Candidate passed the contract's gate |
| `POSITIVE_BELOW_GATE` | Favorable direction, below the material gate |
| `NEGATIVE_UNDER_CONTRACT` | Candidate failed the contract's gate (measured negative) |
| `INCONCLUSIVE` | Evidence could not decide the question |
| `STOP_FOR_COST` | Stopped by a frozen cost or overhead gate |
| `DUPLICATE_EVIDENCE` | Already answered by existing evidence |
| `INFRASTRUCTURE_ONLY` | No science produced (runtime/infra failure or speed work) |
| `READY_FOR_DESKTOP` | Fail-closed transfer package to desktop |
| `CLOSED` | Closed without a candidate verdict (typically the same-job **control/reference** arm; effects require a `CLOSED` reference) |

### `decision.facts` (`molgap-closure-facts-v1`)

Allowed keys (`molgap-closure-facts-v1`, `validate_closure_facts`): `format`, `failure_mode`, `search_terms`, `effect_meV`, `ci_meV`, `source`.
Rules: `format == "molgap-closure-facts-v1"`; `failure_mode` in `FAILURE_MODES` or `null`; `search_terms` <=8 strings or `null`; `effect_meV` finite number or `null`; `ci_meV` `[low, high]` or `null`; `source` repo path or `null`.
Non-null `effect_meV` requires reference arm closed `CLOSED` and exact `pairs[]` (`delta`, `ci95`) + prediction hash match in `source`; otherwise `effect_meV`/`ci_meV`/`source` must be `null`. `draft-closure` fills measured fields and leaves `failure_mode`/`search_terms` `null`.

`failure_mode` enum (`FAILURE_MODES`, enforced by `lookup --failure-mode`):

| Value | Meaning |
|---|---|
| `infra_failure` | Runtime or infrastructure failure; no usable predictions |
| `gate_or_cost_stop` | Frozen overhead/cost gate stopped the run before science |
| `lost_at_scale` | Smaller-scale gain did not hold at larger scale |
| `early_gain_fades` | Candidate leads early and loses by the selected endpoint |
| `endpoint_regression` | Same-job candidate worse at endpoint; paired CI excludes zero |
| `below_gate` | Favorable direction but below the material gate |
| `confounded_change` | Multiple factors changed; cause not isolated |
| `comparator_missing` | No matched comparator exists for the claim |
| `discriminator_missing` | Candidate causes listed, discriminating run not done |
| `boundary_pass` | Passes gate only near its threshold |

### `attribution_label.json` side files

Weak-evidence labels beside a decision file (`trajectory_ids`, `failure_mode` or `null`, optional `secondary_modes`,
`source: "agent_interpretation"`, `basis: {path, sha256}`, `labeled_at`, `note`). Used by `cards` only when the
trajectory has no closure-fact `failure_mode`. Not schema-checked by code; confirm against the decision file.

### Errata

Narrative-only hash-mismatch excuse under `experiments/<q>/errata/*.json` (`format: "molgap-erratum-v1"`,
`artifact_class: "narrative"`, `effect: "narrative_only"`, `target_locator`, `recorded_sha256`, `observed_sha256`,
`what_is_known`, `what_is_not_known`, `author`, `date`, `note`). Limits (`validate._load_errata`):

- Target must be a forward-slash `.md` path inside the same `experiments/<q>/`; both digests must match exactly.
- Applies only to `validate` (reported in `known_gaps`); ignored by strict acceptance, `check`, and closure. Never
  covers checkpoints, predictions, or JSON records.

### Hash classes & derived indexes

| Class | Checked by | Strength |
|---|---|---|
| Bound artifact SHA-256 | `validate`, `replay-pool`/backtests, strict acceptance | Fail-closed (NUL-free files in Git `HEAD` match uniform LF/CRLF; external, NUL, mixed-ending, and `finalization.json` match byte-for-byte) |
| Closure input hashes | `accept_run.py --strict` | Strict; errata do not apply |
| Erratum digest pair | `validate` only | Exact match; narrative only |
| Index freshness (`source_digest`) | `check --frozen` | Strict for derived bytes, not artifact integrity |
| Advisory (`attribution_label.basis`, `analyze` inputs) | None | Reader provenance only |

Never hand-edit `research_memory/derived/` (`trajectory_index.json`, `cost_ledger.json`, `role_reuse_index.json`,
`reference_reuse_index.json`, `screening_backtest.json`, `replay_pool.json`, `policy_backtest.json`,
`completeness_report.json`, `research_summary.json`, `research_summary.md`, `families.md`, `families/`).

## Planning with the evidence

1. Run `cards --outcome NEGATIVE_UNDER_CONTRACT --compact` and `cards --term <word> --compact` to scan prior work.
2. Run `lookup --mechanism <word> --compact`, `lookup --failure-mode <mode> --compact`, and `levers --family <pat>` to
   collect related failures, `unfalsified` families, and `search_terms`.
3. Run `gaps --compact` to spot single-trajectory families or cross-scale contradictions.
4. Open each match's `decision_ref`, `effect_source`, and attribution text — cards are pointers, not primary evidence.
5. Literature-search with `search_terms` + your mechanism terms before proposing a route.
6. Publish a prospective trajectory before any diagnostic or training
   ([Research Protocol](RESEARCH_PROTOCOL.md#new-research-question-protocol)).

Honest limits:
- `effect_meV` is populated only from closure facts or `paired_analysis.json` against a `CLOSED` reference; `null` means
  unrecorded or seed-noise comparison, never zero.
- `mechanism` mixes slugs and free prose; try multiple synonyms when querying.
- `attribution_label` carries `agent_interpretation` (weak evidence) — verify against the decision file.
- `unfalsified` and `gaps` mark thin evidence, not promising hypotheses.

## When verifiers disagree

| Situation | Which one wins |
|---|---|
| `validate` passes with narrative erratum in `known_gaps`, `accept_run.py --strict` returns `BLOCKED` | **Strict recheck (`accept_run.py --strict`)** governs replay, comparison and promotion; `validate` + narrative errata never unblock strict replay |
| `check --frozen` passes, `validate` fails | **`validate`** — index freshness is not artifact integrity |
| `validate` passes, `check --frozen --portable` fails | Both are right; valid locally but not portable (report missing/evicted artifacts honestly) |
| Card or label vs decision file | **The decision file** ([AGENTS precedence](../../AGENTS.md#authority)) |
