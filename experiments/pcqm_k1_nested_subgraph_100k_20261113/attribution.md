# Terminal Attribution: K1 `nested_subgraph` 100K (`pcqm-k1-nested-subgraph-100k-20261113-s42`)

Post-training paired attribution comparing candidate arm `nested_subgraph` (`pcqm-k1-nested-subgraph-100k-20261113-s42`, local RTX 5060, PyTorch `2.7.1`) against the retained external reference `muon` (`pcqm-k1-muon-100k-20261106-s42/v1/version1/muon`, Kaggle Tesla T4, PyTorch `2.11.0`) across `50,000` aligned internal development molecules.

## 1. Mechanism & Footprint

- **Intervention**: Rooted `h=2` induced-subgraph MPNN (`NestedSubgraphMessage`, `hidden_channels=64`, `num_layers=2`) attached at `SlotName.PRE_MESSAGE` layer `4` of `neural_atom_k1` with zero-initialized return projection (`src/molgap/k1_nested_subgraph.py`).
- **Parameter count**: `3,665,747` (`367` tensors, `muon`) -> `3,774,291` (`399` tensors, `nested_subgraph`), `+108,544` parameters (`+2.961%`).
- **Shared frozen contract**: `pcqm_fixed_100k_v1` (`100,000` train / `50,000` development rows, `source_idx_sha256` `3270b30071692d712c468fa52fa247d37f0a8e4b5444dc7c913aadb8ca0832c4`, float32 targets bound via pinned `target_manifest.json` `9c5b2f4a161cc3d6539cfa7bca89da37773276b201d775b1ab88300ac4c9de10`), `k1-muon-adamw-fp32-v1`, `fp32` without TF32, cosine schedule over `40` of `60` passes (`31,240` steps, batch size `128`, seed `42`), `best-development-live`.

## 2. Executive Metrics & Gate Verdict

- **Control MAE (`muon`, selected/final pass 40)**: `0.132521 eV` (`0.1325206676620245`)
- **Candidate MAE (`nested_subgraph`, selected pass 36, step `28,116`)**: `0.130490 eV` (`0.13048968615472317`); final pass 40: `0.131122 eV`
- **Paired Delta (`nested_subgraph - muon`)**: `-0.002031 eV` (`-2.031 meV`), 10,000-resample row bootstrap (seed `42`) 95% CI `[-0.002675, -0.001405] eV` (`[-2.675, -1.405] meV`), `P(candidate better) = 1.0`
- **Material Gate (`<= -3.0 meV`)**: `False` (failure mode: `below_gate`) | **Noninferiority Gate (`<= +0.5 meV`)**: `True`
- **Row Win / Loss / Tie**: `25,742` / `24,257` / `1` (`51.484%` win rate; median row gain `+0.001650 eV`, IQR `0.067718 eV`)
- **Signals**: `POSITIVE_SUB_THRESHOLD`; `global_bias_shift_dominant = False`, `late_horizon_fade = False`, `early_lag_late_recovery = False`

## 3. Signed Bias & Constant-Shift Check

| Arm | Raw MAE (eV) | Mean Signed Error (eV) | Optimal Constant Shift (eV) | Shift-Corrected MAE (eV) |
| --- | ---: | ---: | ---: | ---: |
| `muon` | `0.132521` | `+0.020503` | `-0.013253` | `0.131888` |
| `nested_subgraph` | `0.130490` | `+0.012164` | `-0.006924` | `0.130324` |

- **Shift-corrected delta**: `-0.001564 eV` (`22.98%` of raw delta removed by constant scalar shift). The improvement is not a global bias shift.

## 4. Tail Error Quantiles (eV)

| Quantile | `muon` (eV) | `nested_subgraph` (eV) | Delta (`cand - ctrl`) |
| --- | ---: | ---: | ---: |
| `p50_eV` | `0.085938` | `0.084789` | `-0.001148` |
| `p75_eV` | `0.171932` | `0.169958` | `-0.001975` |
| `p90_eV` | `0.298421` | `0.294629` | `-0.003792` |
| `p95_eV` | `0.407979` | `0.400662` | `-0.007316` |
| `p99_eV` | `0.715794` | `0.701124` | `-0.014669` |
| `max_eV` | `5.132540` | `4.935595` | `-0.196945` |

Tail reductions scale monotonically from median (`-1.15 meV`) through `p95` (`-7.32 meV`) and `p99` (`-14.67 meV`), while top-20 single outliers show mixed local structural trade-offs (for example, `source_idx` `130704` improves by `-923.2 meV`, whereas `148406` worsens by `+1143.1 meV`).

## 5. Gap Regime & Quartile Stratification

### Physical HOMO-LUMO Gap Bins

| Stratum | Rows | Target Mean (eV) | `muon` MAE (eV) | `nested_subgraph` MAE (eV) | Delta (eV) | Cand Win % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `<4.0_eV` | `3,725` | `3.5308` | `0.188239` | `0.182619` | `-0.005620` | `51.76%` |
| `4.0_to_6.0_eV` | `34,837` | `5.0830` | `0.118658` | `0.116961` | `-0.001697` | `51.70%` |
| `6.0_to_8.0_eV` | `10,451` | `6.6951` | `0.151153` | `0.148771` | `-0.002382` | `50.89%` |
| `>=8.0_eV` | `987` | `8.5116` | `0.214242` | `0.217684` | `+0.003442` | `49.34%` |

### Target Quartiles

| Stratum | Rows | Target Mean (eV) | `muon` MAE (eV) | `nested_subgraph` MAE (eV) | Delta (eV) | Cand Win % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Q1_0_25` | `12,527` | `4.1333` | `0.146074` | `0.142914` | `-0.003160` | `52.14%` |
| `Q2_25_50` | `12,505` | `4.9915` | `0.114330` | `0.113273` | `-0.001057` | `50.92%` |
| `Q3_50_75` | `12,478` | `5.5909` | `0.115998` | `0.113836` | `-0.002162` | `52.12%` |
| `Q4_75_100` | `12,490` | `6.7766` | `0.153647` | `0.151904` | `-0.001743` | `50.76%` |

Gains concentrate in conjugated/narrow-gap systems (`<4.0 eV`, `-5.62 meV`) and hold across `Q1..Q4` (`-1.06` to `-3.16 meV`), with regression limited to the sparse `>=8.0 eV` tail (`987` rows, `+3.44 meV`).

## 6. Complementarity & Post-Hoc Ensemble Check

- **Mean `|Pred_Cand - Pred_Ctrl|`**: `0.055824 eV`
- **Signed residual Pearson `r`**: `0.920239` | **Absolute error Pearson `r`**: `0.893570`
- **50/50 equal blend MAE**: `0.128175 eV` (`-2.315 meV` vs `nested_subgraph`, `-4.346 meV` vs `muon`)
- **Grid-oracle blend MAE**: `0.128077 eV` (`w = 0.58`, gain `+2.413 meV` over best single)

Same-role development blending is descriptive only and does not substitute for single-model material gates or held-out evaluation.

## 7. Optimization & Full 40-Pass Trajectory

- **Early window (`passes 1..10`) mean delta**: `+0.000947 eV` (initial warmup lag through pass 7 before crossing over at passes 8-10)
- **Mid window (`passes 11..20`) mean delta**: `-0.000570 eV`
- **Late last 10 passes (`31..40`) mean delta**: `-0.001668 eV` | **Last 5 passes (`36..40`)**: `-0.002074 eV`
- **Candidate ahead in evaluations**: `29 / 40` (including every consecutive evaluation from pass `19` through `40`)
- **Selected vs Final**: `muon` improves through pass 40 (`0.132521` at pass 40, online train `0.076275`); `nested_subgraph` peaks at pass `36` (`0.130490 eV`, online train `0.076143 eV`) and ticks up `+0.632 meV` to `0.131122 eV` at pass `40` (online train `0.074373 eV`).

| Pass | Step | `muon` Dev MAE (eV) | `nested_subgraph` Dev MAE (eV) | Delta (`cand - ctrl`, eV) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 781 | 0.275828 | 0.278029 | +0.002201 |
| 2 | 1562 | 0.225286 | 0.228608 | +0.003322 |
| 3 | 2343 | 0.201920 | 0.201763 | -0.000156 |
| 4 | 3124 | 0.186910 | 0.191386 | +0.004476 |
| 5 | 3905 | 0.183394 | 0.186166 | +0.002772 |
| 6 | 4686 | 0.170508 | 0.173921 | +0.003413 |
| 7 | 5467 | 0.166763 | 0.170568 | +0.003805 |
| 8 | 6248 | 0.162863 | 0.162414 | -0.000449 |
| 9 | 7029 | 0.159682 | 0.157184 | -0.002498 |
| 10 | 7810 | 0.159994 | 0.152578 | -0.007416 |
| 11 | 8591 | 0.154807 | 0.154265 | -0.000542 |
| 12 | 9372 | 0.152330 | 0.150658 | -0.001672 |
| 13 | 10153 | 0.148405 | 0.150184 | +0.001779 |
| 14 | 10934 | 0.152830 | 0.148923 | -0.003907 |
| 15 | 11715 | 0.145499 | 0.145690 | +0.000191 |
| 16 | 12496 | 0.145875 | 0.148019 | +0.002143 |
| 17 | 13277 | 0.147250 | 0.147501 | +0.000251 |
| 18 | 14058 | 0.142581 | 0.146546 | +0.003965 |
| 19 | 14839 | 0.142394 | 0.138415 | -0.003979 |
| 20 | 15620 | 0.143161 | 0.139229 | -0.003932 |
| 21 | 16401 | 0.140600 | 0.140450 | -0.000150 |
| 22 | 17182 | 0.140031 | 0.139926 | -0.000105 |
| 23 | 17963 | 0.138233 | 0.136503 | -0.001730 |
| 24 | 18744 | 0.137403 | 0.136449 | -0.000955 |
| 25 | 19525 | 0.136628 | 0.134978 | -0.001650 |
| 26 | 20306 | 0.135460 | 0.133860 | -0.001600 |
| 27 | 21087 | 0.134629 | 0.134280 | -0.000349 |
| 28 | 21868 | 0.136154 | 0.133692 | -0.002462 |
| 29 | 22649 | 0.133274 | 0.132620 | -0.000654 |
| 30 | 23430 | 0.134460 | 0.132718 | -0.001742 |
| 31 | 24211 | 0.134442 | 0.133367 | -0.001075 |
| 32 | 24992 | 0.133580 | 0.132326 | -0.001254 |
| 33 | 25773 | 0.133196 | 0.131778 | -0.001418 |
| 34 | 26554 | 0.132676 | 0.131250 | -0.001426 |
| 35 | 27335 | 0.132842 | 0.131701 | -0.001141 |
| 36 | 28116 | 0.133279 | 0.130490 | -0.002789 |
| 37 | 28897 | 0.133239 | 0.130847 | -0.002391 |
| 38 | 29678 | 0.133121 | 0.131416 | -0.001704 |
| 39 | 30459 | 0.133040 | 0.130954 | -0.002086 |
| 40 | 31240 | 0.132521 | 0.131122 | -0.001399 |

## 8. Cost, Provenance & Limitations

- **Candidate process wall**: `4409.180094899999 s` (`NVIDIA GeForce RTX 5060`, PyTorch `2.7.1`, `process_wall`, measured); `device_seconds` missing (`requires platform allocation accounting`).
- **Control process wall**: `11177.999006928 s` (`Tesla T4`, PyTorch `2.11.0`, `process_wall`, measured); `device_seconds` missing.
- **Cross-hardware wall ratio (`0.3945x`)**: Reflects hardware differences (`RTX 5060` vs `Tesla T4`) and is not an architectural speedup claim.
- **Limitations**: Single seed (`42`), row bootstrap measures row uncertainty rather than seed variance, development set was selection-used, and candidate vs external reference crosses runtime/PyTorch versions without a same-job `comparison_prelaunch` bundle.
- **Artifacts**: `D:\molgap-artifacts\runs\pcqm-k1-nested-subgraph-100k-20261113-s42\v1\reports\{analysis_summary.json,attribution_report.md,development_curves.png}` (`analysis_summary.json` sha256 `636206fdcc172ac0d33995ac71198bdcc530e89e0f6159525ee11d1883023319`).
