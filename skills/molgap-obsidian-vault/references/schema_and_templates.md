# MolGap Obsidian Vault Schemas & Templates

All hand-written files and YAML registry entries are checked by
`D:/molgap-artifacts/molgap-vault/_scripts/validate_vault.py`.

## 1. `_data/papers.yaml` Schema

Top-level key: `papers:` (list of mappings). Required keys on every entry:

| Key | Requirement / Allowed Values |
|---|---|
| `id` | Unique, starts with `paper_` (e.g., `paper_transformer_m_2023`) |
| `title` | Non-empty full paper title |
| `short_name` | Compact label shown in dashboards & cards |
| `authors_year` | Author & year citation |
| `venue` | Conference, journal, or arXiv identifier |
| `category` | `backbone` \| `pair_interaction` \| `topology_pe` \| `optimizer` \| `readout_loss` \| `3d_legacy` \| `3d_supervision` \| `pretraining_multitask` |
| `empirical_signal` | `proven_winner` \| `high_potential` \| `mixed_scale_limited` \| `negative_under_contract` \| `candidate_to_explore` |
| `attack_priority` | `P0_核心深挖` \| `P1_重点攻坚` \| `P2_参考对照` \| `P3_已证伪避坑` |
| `absorbed` | What MolGap absorbed or plans to test |
| `attack_notes` | Concise empirical justification & next test |

## 2. `_data/mechanisms.yaml` Schema

Top-level key: `mechanisms:` (list of mappings). Required keys on every entry:

| Key | Requirement / Allowed Values |
|---|---|
| `id` | Unique, starts with `mech_` |
| `title` | Concise Chinese + English tag |
| `papers` | Non-empty list of existing `paper_*` IDs from `_data/papers.yaml` |
| `status` | Prefix status string (`✅ 主力核心`, `🔥 强潜力攻坚`, `⚠️ 待跨架构验证`, `🧭 待立项调研`, `❌ 已证伪`) |
| `description` | Mechanism placement, complexity, and empirical summary |

## 3. Hand-Written Note Format (`notes/`)

- **Path**: `D:/molgap-artifacts/molgap-vault/notes/YYYY-MM-DD_<lowercase_slug>.md`
- **Filename regex**: `^\d{4}-\d{2}-\d{2}_[a-z0-9][a-z0-9_-]*\.md$`
- **Required frontmatter**:
  - `title` (non-empty string)
  - `kind`: `research_note` | `experiment_review` | `architecture_idea` | `decision_memo`
  - `created`: `YYYY-MM-DD`
  - `updated`: `YYYY-MM-DD`
  - `status`: `active` | `resolved` | `archived`
  - `related_papers`: list of `paper_*` IDs from `_data/papers.yaml` (or `[]`)
  - `related_models`: list of model IDs (or `[]`)
  - `tags`: list containing `molgap/note` (no `.`, spaces, or leading `#`)
- **Required body structure**: at least one H1 (`# `) and one H2 (`## `), plus a one-line conclusion callout (`> [!abstract] 一句话结论`) and valid Obsidian `[[...]]` links.

## 4. Paper Reading Note Format (`reading/`)

- **Path**: `D:/molgap-artifacts/molgap-vault/reading/<lowercase_slug>.md`
- **Filename regex**: `^[a-z0-9][a-z0-9_-]*\.md$`
- **Required frontmatter**:
  - `title` (non-empty string)
  - `kind`: `paper_reading` (fixed)
  - `paper_id`: existing `paper_*` ID in `_data/papers.yaml`, or `null`
  - `authors_year`: string
  - `venue`: string
  - `leverage_axis`: `3d_supervision` | `higher_order_topology` | `pretraining_multitask` | `optimizer_scaling` | `other`
  - `attack_priority`: `P0_核心深挖` | `P1_重点攻坚` | `P2_参考对照` | `P3_已证伪避坑`
  - `created`: `YYYY-MM-DD`
  - `updated`: `YYYY-MM-DD`
  - `status`: `to_read` | `reading` | `distilled` | `adopted` | `rejected`
  - `tags`: list containing `molgap/reading`
- **Required body structure**: at least one H1 (`# `) and one H2 (`## `), plus pure-2D contract & cheapest 100K falsifier assessment and valid `[[...]]` links.
