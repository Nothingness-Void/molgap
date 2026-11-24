---
name: molgap-obsidian-vault
description: >-
  Maintain, query, validate, and extend the MolGap Obsidian research vault at
  D:/molgap-artifacts/molgap-vault. Use whenever syncing local checkpoints or
  Tail Holdout v1 evaluations, registering new papers or mechanism nodes, writing
  formatted research or paper-reading notes, or querying the unified model,
  paper, and knowledge-graph hub across any agent runtime.
---

# MolGap Obsidian Vault Workflow

The authoritative vault lives at `D:/molgap-artifacts/molgap-vault` (outside
the `d:/文档/molgap` Git repository). Always run scripts with
`d:/文档/molgap/.venv/Scripts/python.exe`.

## 1. Zone Ownership & Hard Constraints

| Path | Owner | Rule |
|---|---|---|
| `models/`, `ensembles/`, `papers/`, `mechanisms/`, `rml_evidence/`, `00_总览看板.md`, `MolGap_*.base`, `MolGap_Knowledge_Graph.canvas`, `.obsidian/graph.json` | Auto-generated | **Read-only.** Regenerated and overwritten by `_scripts/build_vault.py`. Never hand-edit. |
| `_data/papers.yaml`, `_data/mechanisms.yaml` | Structured registries | Add or edit paper/mechanism entries using the strict schema, then rebuild + validate. |
| `notes/`, `reading/`, `_templates/`, `01_评测标尺说明.md`, `02_维护规则.md` | Protected hand-written | Preserved across rebuilds. Follow templates, filename regexes, and frontmatter enums strictly. |
| `.obsidian/{app,appearance,core-plugins,workspace}.json` | User UI settings | **Never modify or overwrite.** |

Hard boundaries:
- **Zero repo side effects:** never modify `d:/文档/molgap` files for vault updates.
- **No cross-scale mixing:** keep `tail_mae_mev`, `dev_mae_mev`, `official_valid_mev`, and `ext_3target_mae_mev` strictly separate; leave unmeasured fields `null`. Never evaluate or score `train_rows > 3,250,000` on `tail_holdout_v1`.
- **No metric fabrication or stale tags:** every metric comes from `D:/molgap-artifacts/` or frozen repo decisions; never use `.` or spaces inside `tags` (use `scale/3250k`, not `scale/3.25M`).

## 2. Standard Workflows

1. **Sync new checkpoints, runs, or `tail_holdout_v1` JSONs**
   - Ensure artifacts reside under `D:/molgap-artifacts/` in their standard paths, then run Section 3. Single models deduplicate by 64-char lowercase `checkpoint_sha256`.
   - The same rebuild generates RML evidence notes. A model card links to one
     trajectory only when the checkpoint SHA matches uniquely. Follow the
     evidence note to the decision, retained training curve, strict replay
     and the GitHub source at the frozen commit. Unbound cards remain unlinked.
   - If a new model family or mechanism variant is introduced, add explicit lineage matching in `_scripts/build_vault.py` (`infer_family_and_variant`) and `_scripts/knowledge_graph.py` (`infer_lineage_for_single` / `infer_lineage_for_ensemble`) without leaking candidate mechanisms onto control arms.
2. **Register a new paper or mechanism node**
   - Never create `papers/*.md` or `mechanisms/*.md` by hand.
   - Append to `_data/papers.yaml` and/or `_data/mechanisms.yaml` using the exact enums in [schema_and_templates.md](references/schema_and_templates.md), then run Section 3.
3. **Write a research / review note (`notes/`)**
   - Filename: `YYYY-MM-DD_<lowercase_slug>.md`.
   - Copy frontmatter & headings from `D:/molgap-artifacts/molgap-vault/_templates/research_note.md`, include tag `molgap/note`, and link nodes via `[[papers/...]]`, `[[mechanisms/...]]`, `[[models/...]]`, or `[[ensembles/...]]`.
4. **Write a paper-reading note (`reading/`)**
   - Filename: `<lowercase_slug>.md`.
   - Copy frontmatter & headings from `D:/molgap-artifacts/molgap-vault/_templates/paper_reading.md`, include tag `molgap/reading`, link `paper_id` to `_data/papers.yaml` (or `null` if candidate not yet registered), then run Section 3 so `00_总览看板.md` and `MolGap_Notes.base` index it.
5. **Fast lookup entrypoints**
   - Start from `00_总览看板.md`, `MolGap_Models.base`, `MolGap_Papers.base`, `MolGap_Notes.base`, and `MolGap_Knowledge_Graph.canvas` (details in [graph_and_scales.md](references/graph_and_scales.md)). Vault is a derived view: verify final authority against remote schedulers, `CURRENT_STATE.md`, and RML records when making promotion or training decisions.

## 3. Mandatory Finish Gate

After **any** change to `_data/`, `_scripts/`, `notes/`, or `reading/`, run both commands and require exit code `0` before finishing:

```powershell
d:\文档\molgap\.venv\Scripts\python.exe D:\molgap-artifacts\molgap-vault\_scripts\build_vault.py
d:\文档\molgap\.venv\Scripts\python.exe D:\molgap-artifacts\molgap-vault\_scripts\validate_vault.py
```
