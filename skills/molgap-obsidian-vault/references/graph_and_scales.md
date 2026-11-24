# Evaluation Scales, Graph View & Lineage Maintenance

## 1. Four Strictly Isolated Evaluation Scales

Never subtract, convert, or merge MAE values across these four scales:

| Field | Dataset / Scope | Rule |
|---|---|---|
| `tail_mae_mev` | `tail_holdout_v1` (rows 3,250,000..3,378,605; 128,606 mols) | Unified cross-run benchmark. Rejected if `train_rows > 3,250,000` (`overlaps_tail`). Runs ~7-8 meV higher than 50K dev for the same model. |
| `dev_mae_mev` | Internal 50K Dev (`ema_0.999`, `ema_0.9999`, or `live`) | Valid only for matched paired A/B comparisons within the same split and `dev_metric_kind`. |
| `official_valid_mev` | OGB-LSC PCQM4Mv2 `valid` (73,545 mols) | Report-only after candidate freeze. Never use for tuning or checkpoint selection. |
| `ext_3target_mae_mev` | Track A External 1,973 B3LYP (HOMO/LUMO/Gap avg) | Track A production scale only; completely separate from Track B PCQM4Mv2 single-Gap models. |

## 2. Obsidian Graph View (`Ctrl+G`) Filters

`.obsidian/graph.json` pre-configures 11 color groups (Green P0, Gold P1, Red P3, Orange P2, Cyan mechanisms, Purple ensembles/Track A, Rose-Pink hand-written `notes/` + `reading/`, Blue `scale/3250k`, Yellow `DeMol-CIN`, Light Green `GPTrans-T`, Muted Slate `k1`) and excludes hub files.

Useful search strings:
- Default clean web: `-path:00_总览看板 -path:01_评测标尺说明 -path:02_维护规则 -path:MolGap_ -path:_templates`
- Active routes without retired K1: `-path:00_总览看板 -path:01_评测标尺说明 -path:02_维护规则 -path:MolGap_ -path:_templates -tag:#family/k1`
- Backbone + breakthroughs + notes only: `path:papers OR path:mechanisms OR path:ensembles OR path:notes OR path:reading OR tag:#scale/3250k OR tag:#family/demol_cin`

## 3. Extending Automatic Lineage Inference

When adding a new architecture family or model variant:
1. Update `infer_family_and_variant` in `_scripts/build_vault.py` using training-contract metadata before string fallback, and inspect only the trailing arm segment (`arm_tail`) for run IDs that embed candidate prefixes.
2. Update `infer_lineage_for_single` / `infer_lineage_for_ensemble` in `_scripts/knowledge_graph.py`. Inspect only trailing arm suffixes of logical-run aliases so control arms (such as `pair_update_norm`) never inherit candidate-only mechanisms.
3. Rebuild and validate with both scripts (`build_vault.py` then `validate_vault.py`).
