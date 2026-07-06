# Phase 8: Scaling & Architecture

## Goal
Produce a better **v2 B3LYP base model** than the Phase 7 300k hybrid, or prove
that the Phase 7 model should remain the production base. Phase 8 is still model
optimization; it is **not** the commercial-molecule database build.

Final outcome: `phase8_replacement_hybrid` is selected as the v2 B3LYP base.
The Phase 7 hybrid (`models/hybrid_fusion_optuna.pt`) remains the frozen v1
fallback/control.

## Why Phase 8 changed
The old Phase 8 chemical-space screening work is delivery-layer trust tagging, so
it now belongs in Phase 10. Its useful output still matters: P8.1 quantified the
Phase 7 training space and exposed the coverage gaps. But screening commercial
molecules before selecting the final base model would force Phase 9/10 artifacts
to be rebuilt after every base-model change.

The new Phase 8 therefore targets the only remaining B3LYP-surrogate levers:

1. **training coverage** — fill known sparse regions instead of re-drawing the
   same PubChemQC distribution;
2. **trainable encoders** — frozen-embedding head probes are exhausted;
3. **MoE A/B** — test a router head only after the encoder can adapt.

## P8.1 Training-space characterization (done)
`scripts/phase8/characterize_training_set.py` →
`results/phase8/training_space.json`.

300k PubChemQC molecules, CHONSFCl, MW 200-1000:

- Elements: C/H plus N 94%, O 90%, S 33%, Cl 20%, F 19%.
- MW: median 326, p99 709; very large molecules are rare.
- Labels: Gap median 4.81 eV, p1-p99 2.90-7.42 eV.
- Topology sample: aromatic rings median 2, p99 5; aromatic-atom fraction median
  0.43, p99 0.84; rotatable bonds median 5, p99 16.

Coverage gaps called out by `CURRENT_STATE.md`:

- high conjugation / high aromatic fraction;
- narrow gap / charge-transfer-like molecules;
- low S/Cl coverage relative to useful commercial chemistry.

## P8.2 Sampling spec (done, 2026-06-25)
Defined a broader-coverage PubChemQC refetch that fills sparse bins rather than
repeating the Phase 7 distribution. The intended dataset is **not** old300k plus
top-up. It is a fixed-size replacement set:

```
phase8_replacement_300k = phase7_300k - N easy/common rows + N targeted hard rows
```

This keeps total training size fixed at 300k, so the controlled variable is
coverage distribution rather than dataset size.

Artifacts:

- descriptor cache: `results/phase8/training_gap_descriptors.csv`
- executable spec: `results/phase8/sampling_spec.json`
- readable spec: `results/phase8/sampling_spec.md`
- fetcher: `scripts/phase8/fetch_targeted_topup.py`
- replacement assembler: `scripts/phase8/assemble_replacement_dataset.py`
- smoke CSV: `data/raw/phase8_targeted_topup_smoke.csv`
- availability probe CSV: `data/raw/phase8_targeted_topup_probe.csv`

Current Phase 7 coverage gaps from the full 300k descriptor cache:

| region | count | fraction |
|---|---:|---:|
| Gap `<2.5 eV` | 912 | 0.30% |
| Gap `<3.0 eV` | 4,041 | 1.35% |
| aromatic rings `>=5` | 5,045 | 1.68% |
| aromatic atom fraction `>=0.80` | 6,204 | 2.07% |
| aromatic edge (`rings>=5` or `frac>=0.85`) | 6,639 | 2.21% |
| MW `>=500` | 19,637 | 6.55% |
| MW `>=700` | 3,285 | 1.10% |
| S/Cl hard subset | 25,443 | 8.48% |
| flexible hard subset | 9,405 | 3.14% |

Recommended quota axes:

| axis | sparse target | reason |
|---|---|---|
| gap | low-gap tail, especially `<3 eV` | known B3LYP/model blind spot; OLED-like region |
| aromaticity | `aromatic_rings >=5` or `aromatic_atom_fraction >=0.8` | high-conjugation edge |
| elements | S-containing and Cl-containing molecules | underrepresented but allowed elements |
| size | MW `500-1000`, especially `>700` | p99 of v1 is only ~709 despite max 1000 |

The executable 200k top-up priority buckets are:

| priority | bucket | quota |
|---:|---|---:|
| 1 | `very_low_gap` | 30,000 |
| 2 | `low_gap_aromatic_edge` | 40,000 |
| 3 | `large_aromatic_edge` | 26,000 |
| 4 | `very_large_general` | 20,000 |
| 5 | `s_or_cl_hard` | 20,000 |
| 6 | `aromatic_edge_general` | 18,000 |
| 7 | `flexible_hard` | 10,000 |
| 8 | `large_mw_500_700` | 36,000 |

The fetcher excludes Phase 7 CIDs/canonical SMILES, preserves the hard element
set `{C,H,N,O,S,F,Cl}`, and supports `--resume` for long runs. A 10-file / 2-window availability probe produced 597
targeted candidates; common large/SCl/aromatic buckets fill quickly, while
`very_low_gap` and `low_gap_aromatic_edge` are genuinely rare and require a long
scan.

Rare-first scan result:

- command class: `--include-buckets very_low_gap low_gap_aromatic_edge`
- scanned: 40 HF files x 4 random windows/file x 30 MB
- output: `data/raw/phase8_targeted_topup_rare_probe.csv`
- rows: 675 total = 402 `very_low_gap` + 273 `low_gap_aromatic_edge`
- report: `results/phase8/rare_probe_report.json`
- gap bins: 402 `<2.5 eV`, 196 `2.5-3.0 eV`, 77 `3.0-3.2 eV`

Balanced non-rare probe:

- command class: exclude rare buckets; collect large/aromatic/SCl/flexible buckets
- scanned: 30 HF files x 2 random windows/file x 20 MB
- output: `data/raw/phase8_targeted_topup_balanced_probe.csv`
- rows: 3,173 total in 272 s (`11.7 rows/s`)
- report: `results/phase8/balanced_probe_report.json`

Decision: the original 30k/40k low-gap quotas are **availability ceilings, not
hard requirements**. Low-gap chemistry is genuinely sparse in PubChemQC under the
current CHONSFCl/MW filters. Do not waste hours trying to force 70k rare rows.
Keep rare rows as a dedicated hard slice and use sampling/loss weights during
training; fill the rest of the top-up with the easier hard-coverage buckets.

Replacement assembler smoke:

- inputs: `data/raw/phase8_targeted_topup_rare_probe.csv` +
  `data/raw/phase8_targeted_topup_balanced_probe.csv`
- usable candidates: 3,847
- output: `data/raw/phase8_replacement_300k_probe.csv`
- output rows: 300,000 exactly
- report: `results/phase8/replacement_probe_report.md`

Probe coverage shift after replacing 3,847 easy/common rows:

| flag | old fraction | replacement fraction | delta n |
|---|---:|---:|---:|
| low-gap (`gap < 3.2`) | 2.44% | 2.73% | +876 |
| large (`MW >= 500`) | 6.55% | 7.35% | +2,406 |
| aromatic edge | 2.21% | 2.46% | +744 |
| S/Cl hard | 8.48% | 9.05% | +1,721 |
| flexible hard | 3.14% | 3.42% | +855 |
| any P8 hard | 15.60% | 16.88% | +3,847 |

Historical next step from this probe was to fetch enough replacement candidates
for a first real cut and then assemble `data/raw/phase8_replacement_300k.csv`.
That step is now complete. The interrupted `phase8_targeted_topup_200k.csv` is
diagnostic only and should not be treated as the final replacement candidate pool.

## P8.2c Replacement 300k first cut (done, 2026-06-25)

Formal fixed-size replacement dataset:

- output: `data/raw/phase8_replacement_300k.csv`
- report: `results/phase8/replacement_dataset_report.md`
- old rows removed: 38,620 easy/common Phase 7 rows
- targeted rows inserted: 38,620 hard replacement candidates
- final rows: 300,000 exactly
- duplicate canonical SMILES: 0
- target NaNs: 0
- `gap <= 0`: 0

Targeted rows used:

| bucket | n |
|---|---:|
| `large_mw_500_700` | 13,847 |
| `s_or_cl_hard` | 7,677 |
| `very_large_general` | 4,593 |
| `aromatic_edge_general` | 4,185 |
| `flexible_hard` | 3,842 |
| `large_aromatic_edge` | 3,801 |
| `very_low_gap` | 402 |
| `low_gap_aromatic_edge` | 273 |

Coverage shift vs Phase 7 control:

| flag | old fraction | replacement fraction | delta n |
|---|---:|---:|---:|
| low-gap (`gap < 3.2`) | 2.44% | 3.61% | +3,510 |
| large (`MW >= 500`) | 6.55% | 15.04% | +25,492 |
| aromatic edge | 2.21% | 5.21% | +8,984 |
| S/Cl hard | 8.48% | 13.01% | +13,581 |
| flexible hard | 3.14% | 6.46% | +9,956 |
| any P8 hard | 15.60% | 28.47% | +38,620 |

Historical next step from this cut was to build sharded 2D + 3D ETKDG graph
caches for `data/raw/phase8_replacement_300k.csv`, leaving all Phase 7
data/caches intact for the same-size control comparison. That graph-cache build
is now complete in P8.5.

## P8.3 30k MoE decision experiment (done, 2026-06-25)

Before spending a full 300k retrain on MoE, run a controlled 30k decision
experiment:

1. old30k = first 30k rows from the Phase 7 control dataset;
2. replacement30k = first 30k rows from `phase8_replacement_300k.csv`;
3. train GPS + SchNet encoders separately on each 30k set;
4. align 2D/3D embeddings by `source_idx`;
5. train single `FusionHead` and `MoEFusionHead` on the same embedding/split.

Artifacts:

- runner: `scripts/phase8/run_30k_moe_ab.py`
- graph builder: `scripts/phase8/build_replacement_graphs.py`
- encoder trainer/extractor: `scripts/phase8/train_encoder.py`
- fusion A/B: `scripts/phase8/train_moe_fusion.py`
- summary: `results/phase8/moe_ab_30k_summary.json`

Graph build:

| dataset | 2D graphs | 3D ETKDG graphs | ETKDG failures |
|---|---:|---:|---:|
| old30k | 30,000 | 29,985 | 15 |
| replacement30k | 30,000 | 29,973 | 27 |

Encoder internal test MAE, average over HOMO/LUMO/Gap:

| dataset | GPS 2D avg MAE | SchNet 3D avg MAE |
|---|---:|---:|
| old30k | 0.1543 | 0.1737 |
| replacement30k | 0.1659 | 0.1756 |

These internal test numbers are **not** a direct old-vs-replacement quality
comparison because the replacement split is intentionally harder. They are only
a sanity check that both 30k encoders trained normally.

Fusion A/B on each dataset:

| dataset | head | avg MAE | Gap MAE | delta vs single |
|---|---|---:|---:|---:|
| old30k | single FusionHead | 0.12649 | 0.14774 | — |
| old30k | MoE(4) | 0.12646 | 0.14751 | -0.00003 avg / -0.00022 Gap |
| replacement30k | single FusionHead | 0.13838 | 0.16251 | — |
| replacement30k | MoE(4) | 0.13778 | 0.16211 | -0.00060 avg / -0.00040 Gap |

Conclusion: MoE is a **tie-level gain** at 30k, matching the earlier frozen
Phase 7 result. It is not worth prioritizing a full 300k MoE run. If Phase 8
continues, the next controlled lever should be replacement-data coverage with a
single fusion head, evaluated on a common OOD/hard set.

### End-to-end MoE pilot (done, 2026-06-25)

Question: can the 30k MoE be trained end-to-end, with gradients flowing through
GPS 2D + SchNet 3D + MoE instead of freezing encoder embeddings?

Implementation:

- reusable wrapper: `src/molgap/hybrid.py` (`EndToEndHybrid`)
- trainer: `scripts/phase8/train_end_to_end_hybrid.py`
- data: `replacement30k`, aligned by `source_idx`
- head: `MoEFusionHead`, 4 experts
- training: batch 64, max 60 epochs, patience 10, lr `2e-4`

Result:

| run | best val MAE | test avg MAE | test Gap MAE |
|---|---:|---:|---:|
| replacement30k end-to-end MoE | 0.14362 | 0.14170 | 0.17301 |
| replacement30k frozen-embedding MoE | 0.1387 val | 0.13778 | 0.16211 |

Conclusion: end-to-end MoE is technically feasible on the RTX 5060 at 30k, but
this first run does **not** beat the simpler train-encoders-then-fusion setup.
If revisited, compare against an end-to-end single-head baseline and tune
learning-rate groups / warm-starts before considering a full 300k run.

## P8.4 Common evaluation (done, 2026-06-25)

Before a full 300k retrain, compare the 30k old/replacement models on a **common**
evaluation set. This isolates data coverage value; internal test splits cannot do
that because their distributions differ.

Artifacts:

- evaluator: `scripts/phase8/common_eval_30k.py`
- metrics: `results/phase8/common_eval_30k_metrics.json`
- predictions: `results/phase8/common_eval_30k_predictions.csv`
- summary: `results/phase8/common_eval_30k_summary.md`

Common-eval set:

| slice | valid molecules |
|---|---:|
| Phase 7 OOD-1000 | 999 |
| P8 targeted hard slice | 981 |
| total | 1,980 |

Hybrid results, replacement30k minus old30k:

| eval set | avg MAE delta | Gap MAE delta |
|---|---:|---:|
| all | -0.00216 | -0.00102 |
| Phase 7 OOD-1000 | +0.00033 | +0.00213 |
| P8 targeted hard | -0.00469 | -0.00422 |

Conclusion: the replacement distribution is not a broad OOD breakthrough at 30k,
but it is directionally positive on the hard chemistry Phase 8 targeted. This is
enough to justify one full replacement300k standard hybrid run if compute budget
is available. Keep the next full run to the single `FusionHead`; MoE remains
deprioritized.

### Intermediate-layer fusion pilot (done, 2026-06-25)

Question: can a cheap head-only upgrade improve fusion by concatenating
intermediate GPS/SchNet pooled embeddings instead of using only the final encoder
embedding?

Implementation:

- encoder APIs: `GPSWrapper.encode_layers(...)`,
  `SchNetWrapper.encode_layers(...)`
- layer choice: 2 / 4 / final for each encoder
- trainer: `scripts/phase8/train_layer_fusion.py`
- common evaluator: `scripts/phase8/eval_layer_fusion_common.py`
- comparison table: `results/phase8/intermediate_layer_fusion_comparison.md`

B3LYP replacement30k internal test:

| head | test avg MAE | test Gap MAE | delta avg vs single | delta Gap vs single |
|---|---:|---:|---:|---:|
| single FusionHead | 0.13838 | 0.16251 | - | - |
| MoE(4) | 0.13778 | 0.16211 | -0.00060 | -0.00040 |
| intermediate-layer fusion | 0.13719 | 0.16149 | -0.00118 | -0.00102 |

Common eval, intermediate-layer fusion minus single replacement30k:

| eval set | avg MAE delta | Gap MAE delta |
|---|---:|---:|
| all | +0.00059 | -0.00028 |
| Phase 7 OOD-1000 | -0.00163 | -0.00400 |
| P8 targeted hard | +0.00292 | +0.00359 |

Conclusion: intermediate-layer fusion is a real internal-test gain and improves
the Phase 7 OOD-1000 slice, but it worsens the P8 targeted hard slice. Keep it as
a cheap head-only follow-up after full replacement300k embeddings exist; do not
delay the standard single-head full run for it.

Original P7 300k check: with the historical 2D/3D alignment index
(`results/phase7/align_2d_idx.pt`), intermediate-layer fusion ties/slightly loses
to the ordinary Phase 7 FusionHead (avg/GAP 0.06740/0.07594 vs
0.06711/0.07563). This confirms it is not a P7 baseline replacement. Full table:
`results/phase8/phase7_300k_baseline_lora_layer_comparison.md`.

## P8.5 Full replacement300k graph cache (done, 2026-06-26)
Build full 2D + 3D ETKDG graphs for the broader-coverage data with the same
conformer method as Phase 7 inference. Use sharded streaming writes; do not mix
PM6 training coords with ETKDG inference.

Artifacts:

- graph report: `results/phase8/graph_build_report.json`
- 2D graphs: `results/phase8/pyg_2d_graphs_bond_replacement_300k.pt`
- 3D ETKDG graphs: `results/phase8/pyg_3d_graphs_etkdg_replacement_300k.pt`

| kind | processed | graphs | failed | elapsed |
|---|---:|---:|---:|---:|
| 2D | 300,000 | 300,000 | 0 | 4.0 min |
| 3D ETKDG | 300,000 | 298,957 | 1,043 | 59.0 min |

## P8.6 Full replacement300k standard hybrid (done, 2026-06-26)

Run the default full model path: warm-start GPS/SchNet from Phase 7 checkpoints,
then train the standard single `FusionHead`. MoE is not run at full scale.

Artifacts:

- GPS checkpoint: `models/phase8_gps_replacement_300k.pt`
- SchNet checkpoint: `models/phase8_schnet_replacement_300k.pt`
- Hybrid checkpoint: `models/phase8_hybrid_fusion_replacement_300k.pt`
- summary: `results/phase8/full_replacement_300k_summary.md`
- common eval: `results/phase8/full_replacement_common_eval_metrics.json`

Internal replacement300k test:

| model | best val MAE | best epoch | test avg MAE | test Gap MAE |
|---|---:|---:|---:|---:|
| GPS 2D | 0.10880 | 0 | 0.10870 | 0.13053 |
| SchNet 3D | 0.12230 | 3 | 0.12342 | 0.14842 |
| Hybrid FusionHead | 0.09661 | 49 | 0.09745 | 0.11503 |

Internal splits differ from Phase 7 and are not the model-selection criterion.
The decisive comparison is the shared common eval:

| eval set | P7 avg MAE | replacement avg MAE | delta avg | P7 Gap MAE | replacement Gap MAE | delta Gap |
|---|---:|---:|---:|---:|---:|---:|
| all | 0.14529 | 0.12839 | -0.01690 | 0.17930 | 0.15610 | -0.02320 |
| Phase 7 OOD-1000 | 0.12431 | 0.12144 | -0.00287 | 0.14881 | 0.14479 | -0.00402 |
| P8 targeted hard | 0.16671 | 0.13548 | -0.03123 | 0.21044 | 0.16765 | -0.04279 |

Conclusion: replacement300k standard FusionHead is the selected v2 base. It
improves the broad common eval, slightly improves the Phase 7 OOD-1000 slice,
and strongly improves the P8 targeted hard slice. Full MoE remains unjustified.

## P8.7 Model selection
### Final decision (done, 2026-06-27)

Select `phase8_replacement_hybrid` as the v2 B3LYP base. Keep `phase7_hybrid`
as the frozen v1 fallback and historical control. Decision record:
`results/phase8/v2_selection_decision.md`.

### PCQM4Mv2 valid proxy audit (done, 2026-06-27)

This repeats the Phase 7-era PCQM4Mv2 valid sounding as a coverage stress test,
not as an OGB leaderboard submission. The audit script uses the official valid
split, removes any molecule present in either the Phase 7 or replacement300k
training CSV, keeps CHONSFCl / MW 200-1000, samples 3000 molecules with seed 42,
and evaluates P7 and P8 on the same ETKDG-valid common subset.

Artifacts:

- script: `scripts/phase8/eval_pcqm4mv2_proxy.py`
- metrics: `results/phase8/pcqm4mv2_proxy_p7_vs_p8_metrics.json`
- predictions: `results/phase8/pcqm4mv2_proxy_p7_vs_p8_predictions.csv`

Overall:

| model | common n | Gap MAE | median abs err |
|---|---:|---:|---:|
| Phase 7 hybrid | 2,988 | 0.25444 | 0.17239 |
| replacement300k hybrid | 2,988 | 0.24645 | 0.16939 |
| delta P8 - P7 | - | -0.00798 | -0.00300 |

By nearest-neighbor similarity to the Phase 7 training set:

| P7 train sim bin | n | P7 Gap MAE | P8 Gap MAE | delta P8 - P7 |
|---|---:|---:|---:|---:|
| [0.0,0.3) | 182 | 0.52733 | 0.49857 | -0.02876 |
| [0.3,0.4) | 581 | 0.31846 | 0.29762 | -0.02084 |
| [0.4,0.5) | 945 | 0.22933 | 0.22353 | -0.00581 |
| [0.5,0.6) | 746 | 0.21355 | 0.21055 | -0.00300 |
| [0.6,1.0) | 534 | 0.19331 | 0.19560 | +0.00229 |

Conclusion: this confirms the intended P8.1 story. The replacement data is not
moving the model toward an OGB leaderboard-style regime; it mostly improves the
low-similarity chemistry that the Phase 7 300k set under-covered. The gain is
smaller than the targeted hard common-eval gain but directionally consistent.

### Error-mode audit (done, 2026-06-27)

Artifacts:

- analysis: `results/phase8/v2_error_mode_analysis.md`
- JSON: `results/phase8/v2_error_mode_analysis.json`
- common-eval worst rows: `results/phase8/v2_common_eval_remaining_worst.csv`
- PCQM proxy worst rows: `results/phase8/v2_pcqm_proxy_remaining_worst.csv`

The remaining common-eval worst cases are mostly flexible, large-conjugated,
S/Cl/F-containing, and narrow-gap molecules. PCQM proxy remaining worst cases
are dominated by radical/open-shell SMILES, which are not the core closed-shell
commercial organic database target. This supports selecting v2 while pushing
remaining method/coverage risk into Phase 9/10 Delta/UQ validation.

## P8.8 Expansion500k candidate (done, 2026-06-28)

After v2 selection, a 500k expansion was tested without replacing the v2 replay
set: keep all 300,000 replacement300k rows and append 200,000 non-duplicate rows
from targeted hard buckets plus general in-domain PubChemQC B3LYP molecules.

Artifacts:

- dataset assembly: `results/phase8/expansion_500k_report.md`
- summary: `results/phase8/full_expansion_500k_summary.md`
- common eval: `results/phase8/full_expansion500k_common_eval_metrics.json`
- registry key: `phase8_expansion_hybrid`

Common-eval hybrid result:

| model | all avg MAE | all Gap MAE | OOD avg MAE | OOD Gap MAE | P8 hard avg MAE | P8 hard Gap MAE |
|---|---:|---:|---:|---:|---:|---:|
| Phase 7 full | 0.14529 | 0.17930 | 0.12431 | 0.14881 | 0.16671 | 0.21045 |
| replacement300k full | 0.12838 | 0.15609 | 0.12144 | 0.14478 | 0.13548 | 0.16765 |
| expansion500k full | 0.10560 | 0.12528 | 0.11373 | 0.13399 | 0.09729 | 0.11638 |

Conclusion: expanding beyond targeted replacement helps. The gain is strongest
on the P8 targeted hard slice but also improves OOD-1000. This is a v3 candidate,
not yet the default loader; Phase 9/10 Delta/UQ must be revalidated against the
chosen base before database work.

## P8.9 Head-swap + SchNet retrain probes on 500k (done, 2026-06-29)

Two follow-up questions on the expansion500k base, both **negative**:

**(a) Head swap (MoE, layer fusion) on 500k embeddings.** Re-tested MoE(4) and
intermediate-layer fusion (GPS/SchNet layers 2/4/-1) on the same 497,578
expansion500k embeddings as the single-head baseline. Both still **tie** the
single FusionHead — MoE avg/GAP -0.00003/-0.00004, layer fusion avg/GAP
-0.00001/-0.00041 eV. MoE's marginal gain *shrank* from 30k (≤0.0006 eV) to 500k
(≤0.00004 eV), the opposite of the "experts need more data to diverge"
hypothesis. Head-swap route is **closed**; production stays on the single
FusionHead. Table: `results/phase8/head_swap_500k_comparison.md`.

**(b) SchNet "non-convergence" retrain.** The v3 expansion500k SchNet leg was
only trained 12 warm-start epochs (its cosine schedule annealed lr→1e-6 by ep11),
which *looked* like under-training (train 0.1329→0.0895, val→0.1180 both still
descending). A 30-epoch warm-start continuation (fresh cosine, lr back to 2.1e-4)
was run to test this. It **failed**: the high re-risen lr destroyed the good
ep11 minimum on the first epoch (val 0.1180→0.1336), and it never recovered —
best val only reached **0.1239 @ep17** vs the original **0.1180**, while train
kept dropping to 0.0707 (clear overfitting, not under-training). Stopped at ep21.
This **falsifies the under-training hypothesis**: the original 12ep checkpoint
`phase8_schnet_expansion_500k.pt` is near its generalization limit on this data
and stays as the v3 SchNet leg. Consistent with the standing diagnosis — the
bottleneck is the **B3LYP label ceiling**, not training time or capacity. Log:
`results/phase8/_schnet_exp500k_30ep.log` (kept as evidence; no model artifact
retained).

## P8.10 B3LYP post-hoc residual/stack probes (done, 2026-07-06)

After the user clarified that the target was the B3LYP surrogate itself, not
LoRA/GW Delta, a lightweight post-hoc B3LYP probe was run on the v3 base. This
does not modify GPS, SchNet, or FusionHead checkpoints.

Setup:

- base: `phase8_expansion_hybrid`;
- fit data: aligned expansion500k split (`398,062 / 49,757 / 49,759`);
- residual features: v3 Hybrid HOMO/LUMO/Gap predictions + 16 lightweight RDKit
  context descriptors;
- stack features: v3 GPS, SchNet, and Hybrid B3LYP outputs + the same descriptors;
- external validation: Phase 8 common eval (`ood1000` + `p8_targeted_hard`).

Common-eval MAE:

| model | HOMO | LUMO | Gap | avg |
|---|---:|---:|---:|---:|
| v3 baseline | 0.0943 | 0.0972 | 0.1253 | 0.1056 |
| constant residual | 0.0943 | 0.0971 | 0.1251 | 0.1055 |
| ridge residual | 0.0940 | 0.0970 | 0.1250 | 0.1053 |
| LightGBM residual | **0.0935** | **0.0968** | **0.1250** | **0.1051** |
| ridge output stack | 0.0952 | 0.0981 | 0.1283 | 0.1072 |
| LightGBM output stack | 0.0960 | 0.1003 | 0.1298 | 0.1087 |

Best external delta is only avg/GAP `-0.00049/-0.00029` eV, below the practical
promotion threshold. Output stacking is worse than the v3 Hybrid baseline.

Decision: **negative**. Do not promote B3LYP residual calibration/output stacking
as a default or repeat this path without a materially different validation
signal. Record: `results/phase8/b3lyp_residual_calibrator_decision.md`.

The same B3LYP-only check also tested tail-aware FusionHead fine-tuning from the
selected v3 FusionHead checkpoint, with GPS/SchNet encoders frozen:

| model | HOMO | LUMO | Gap | avg |
|---|---:|---:|---:|---:|
| v3 baseline | 0.0943 | 0.0972 | 0.1253 | 0.1056 |
| low-gap weighted FusionHead | 0.0940 | 0.0967 | 0.1251 | 0.1053 |
| low-gap + MW weighted FusionHead | **0.0939** | **0.0966** | 0.1252 | **0.1052** |

Best external delta is avg/GAP `-0.00037/-0.00011` eV, also below the practical
threshold. The selected v3 FusionHead remains the B3LYP baseline. Record:
`results/phase8/weighted_fusion_probe_decision.md`.

## P8.11 ETKDG conformer-ensemble inference probe (done, 2026-07-06)

One B3LYP-level route did show a small but real signal: inference-time conformer
averaging. This keeps the trained v3 GPS, SchNet, and FusionHead checkpoints
unchanged and uses only seeded ETKDG+MMFF conformers, so it does not violate the
ETKDG training/inference consistency rule.

Setup:

- base: `phase8_expansion_hybrid`;
- 2D graph: one per molecule;
- 3D graph: up to k ETKDG+MMFF conformers per molecule;
- prediction: run SchNet+Fusion per conformer, average final Hybrid outputs;
- validation: Phase 8 common eval with B3LYP labels.

Common-eval deltas versus stored v3 single-conformer predictions:

| inference | all avg delta | all Gap delta | OOD avg delta | OOD Gap delta | P8 hard avg delta | P8 hard Gap delta |
|---|---:|---:|---:|---:|---:|---:|
| k=4 ETKDG ensemble | -0.00099 | -0.00152 | -0.00143 | -0.00179 | -0.00053 | -0.00124 |
| k=8 ETKDG ensemble | **-0.00116** | **-0.00176** | **-0.00158** | **-0.00209** | **-0.00072** | **-0.00142** |

k=8 common-eval MAE:

| model | HOMO | LUMO | Gap | avg |
|---|---:|---:|---:|---:|
| stored v3 single | 0.0943 | 0.0972 | 0.1253 | 0.1056 |
| k=8 ETKDG ensemble | **0.0933** | **0.0964** | **0.1235** | **0.1044** |

Decision: weak-positive inference candidate. It is the only B3LYP-level probe in
this round that clears the practical threshold, but it costs about 8x 3D
conformer generation/SchNet work and should stay opt-in until speed is
benchmarked. API:
`predict_smiles_batch_hybrid_conformer_ensemble()`. Record:
`results/phase8/v3_conformer_ensemble_k8_decision.md`.

### Original selection rule
Use one fixed split per candidate so the comparisons isolate each lever:

1. trainable encoder + single FusionHead on broader coverage data;
2. same data/split, single FusionHead vs MoE head only if the common-eval result
   shows a reason to revisit MoE;
3. select v2 only if the gain clears noise (`scaffold OOD delta > 2x std`).

If MoE ties, keep the single head and record the negative result. If v2 fails to
beat Phase 7 robustly, keep Phase 7 as production and proceed to Phase 9/10 with
the v1 stack.

## Records
Frozen-encoder probes are already closed:

- MoE on frozen Phase 7 embeddings: no OOD win.
- Descriptor-aware fusion on frozen Phase 7 embeddings: tiny tie-level gain.

See `docs/experiment_moe_experts_2026-06-24.md`.
