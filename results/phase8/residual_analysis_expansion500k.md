# Phase 8 — Residual analysis of the expansion500k hybrid (v3)

Date: 2026-06-30

Script: `scripts/phase8/residual_analysis_expansion500k.py`
Input: `results\phase8\full_expansion500k_common_eval_predictions.csv` (1977 molecules; buckets: {'ood1000': 999, 'p8_targeted_hard': 978}).

## Why this analysis

This checks whether the remaining B3LYP-surrogate residual is broad and uniform,
or concentrated in a small chemistry/geometry tail. It also compares the same
bins across Phase 7, replacement300k, and expansion500k so the conclusion does
not rely on v3 in isolation.

## Headline numbers for v3

| bucket | n | homo_mae | homo_bias | lumo_mae | lumo_bias | gap_mae | gap_bias |
|---|---|---|---|---|---|---|---|
| ood1000 | 999 | 0.1037 | 0.0006 | 0.1035 | -0.0092 | 0.1340 | -0.0127 |
| p8_targeted_hard | 978 | 0.0847 | -0.0101 | 0.0908 | 0.0043 | 0.1164 | 0.0122 |

Gap bias is near-zero at the bucket level. There is no simple global offset to
correct.

## Error concentration

- median Gap abs-err: **0.0766** eV
- mean Gap abs-err: **0.1253** eV
- p90 / p99: **0.3048 / 0.6335** eV
- worst 10% of molecules (197/1977)
  hold **37.7%** of total Gap
  absolute error

The residual is tail-heavy rather than uniform.

## V3 residual bins

### By true Gap

| bin | n | mae | bias |
|---|---|---|---|
| 1-2 | 6 | 0.6486 | 0.6465 |
| 2-3 | 95 | 0.1700 | 0.1108 |
| 3-4 | 440 | 0.1225 | 0.0346 |
| 4-5 | 877 | 0.1133 | -0.0105 |
| 5-6 | 425 | 0.1252 | -0.0327 |
| >6 | 134 | 0.1578 | -0.0539 |

### By molecular weight

| bin | n | mae | bias |
|---|---|---|---|
| <300 | 465 | 0.1251 | -0.0136 |
| 300-400 | 386 | 0.1163 | -0.0142 |
| 400-500 | 455 | 0.1263 | -0.0128 |
| 500-600 | 283 | 0.1148 | 0.0022 |
| 600-800 | 336 | 0.1316 | 0.0329 |
| >800 | 52 | 0.2007 | 0.1006 |

### By heteroatom presence

| flag | value | n | gap_mae |
|---|---|---|---|
| has_S | 1 | 684 | 0.1262 |
| has_S | 0 | 1293 | 0.1248 |
| has_Cl | 1 | 407 | 0.1220 |
| has_Cl | 0 | 1570 | 0.1261 |

S/Cl flags are flat, so the earlier low-S/Cl coverage issue is no longer the
dominant bottleneck.

## P7 vs v2 vs v3 by hard bins

### Gap bins

| bin | n | Phase 7_gap_mae | replacement300k_gap_mae | expansion500k_gap_mae | v3_minus_v2 | v3_minus_p7 |
|---|---|---|---|---|---|---|
| 1-2 | 6 | 1.2848 | 1.0219 | 0.6486 | -0.3733 | -0.6362 |
| 2-3 | 95 | 0.3678 | 0.2750 | 0.1700 | -0.1050 | -0.1978 |
| 3-4 | 440 | 0.1985 | 0.1671 | 0.1225 | -0.0446 | -0.0760 |
| 4-5 | 877 | 0.1619 | 0.1445 | 0.1133 | -0.0312 | -0.0486 |
| 5-6 | 425 | 0.1425 | 0.1308 | 0.1252 | -0.0056 | -0.0173 |
| >6 | 134 | 0.1634 | 0.1532 | 0.1578 | 0.0046 | -0.0056 |

### Molecular-weight bins

| bin | n | Phase 7_gap_mae | replacement300k_gap_mae | expansion500k_gap_mae | v3_minus_v2 | v3_minus_p7 |
|---|---|---|---|---|---|---|
| <300 | 465 | 0.1428 | 0.1366 | 0.1251 | -0.0115 | -0.0177 |
| 300-400 | 386 | 0.1473 | 0.1330 | 0.1163 | -0.0167 | -0.0310 |
| 400-500 | 455 | 0.1875 | 0.1605 | 0.1263 | -0.0342 | -0.0612 |
| 500-600 | 283 | 0.1806 | 0.1591 | 0.1148 | -0.0443 | -0.0658 |
| 600-800 | 336 | 0.2278 | 0.1826 | 0.1316 | -0.0510 | -0.0962 |
| >800 | 52 | 0.3514 | 0.2752 | 0.2007 | -0.0745 | -0.1507 |

Important nuance: v3 already improves the hard bins substantially. For example,
Gap 2-3 eV improves from replacement300k to expansion500k, and MW>800 also
improves. The remaining tail is therefore not evidence that targeted expansion
failed; it is evidence that the next B3LYP-only round has lower expected ROI.

## Training-set coverage check

| csv | n | gap_lt_3_n | gap_lt_3_pct | mw_gt_800_n | mw_gt_800_pct | gap_p01 | gap_p05 | gap_p50 | gap_p99 |
|---|---|---|---|---|---|---|---|---|---|
| data\raw\phase8_replacement_300k.csv | 300000 | 6324 | 2.1100 | 2769 | 0.9200 | 2.7080 | 3.3250 | 4.7100 | 7.4340 |
| data\raw\phase8_expansion_500k.csv | 500000 | 16305 | 3.2600 | 9393 | 1.8800 | 2.5660 | 3.1670 | 4.5470 | 7.2870 |

Expansion500k increased both low-gap and very-large molecule coverage versus
replacement300k. The remaining residual persists despite that broader coverage.

## Worst Gap offenders

Full table: `results\phase8\residual_analysis_expansion500k_worst.csv`.

| eval_set | cid | mw | gap | expansion500k_full_hybrid_gap | gap_abserr | n_rotatable | n_aromatic_rings | n_S | n_Cl | n_N | n_O | canonical_smiles |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p8_targeted_hard | 25143081 | 756.7900 | 1.5800 | 4.4300 | 2.8500 | 9 | 1 | 0 | 0 | 0 | 15 | COC(=O)CC1C2(C)CC34OC5(C)OC67C(C(C(=O)C(C)C)C(=O)OC6C3(OC(C)=O)C2OC(C)=O)C(C)(C(OC(C)=O)c2ccoc2)CCC7(O5)C14C |
| p8_targeted_hard | 11263288 | 985.1800 | 2.5800 | 4.2100 | 1.6400 | 20 | 7 | 0 | 0 | 6 | 6 | O=C(O)[C@H](Cc1ccccc1C[C@H](N=C(c1ccccc1)c1ccccc1NC(=O)[C@@H]1CCCN1Cc1ccccc1)C(=O)O)N=C(c1ccccc1)c1ccccc1NC(=O)[C@@H]1CCCN1Cc1ccccc1 |
| p8_targeted_hard | 57417452 | 805.0500 | 3.0600 | 4.4200 | 1.3500 | 3 | 0 | 0 | 0 | 0 | 12 | CCC1/C=C\C=C/CC(C)C(O)C(C)(O)C(=O)C(C)C(O)C(C)C(=O)C(C)C(O)C(C)/C=C\C(=O)OC2C(C)C(CC1)OC1(OC(CC(C)O)C(C)CC1=O)C2C |
| ood1000 | 55542072 | 447.4300 | 3.5800 | 4.7900 | 1.2100 | 8 | 2 | 1 | 0 | 3 | 5 | COc1ccc(NC(=O)CNc2cccc(OC(F)(F)F)c2)cc1S(=O)(=O)N(C)C |
| p8_targeted_hard | 19826067 | 561.6300 | 2.8100 | 3.9500 | 1.1400 | 9 | 4 | 0 | 0 | 3 | 7 | COc1cc2onc(CCCN3CCN(C(c4ccccc4)c4ccccc4)CC3)c2cc1OC.O=C(O)C(=O)O |
| p8_targeted_hard | 25193465 | 533.7700 | 3.5300 | 4.5200 | 0.9900 | 9 | 4 | 3 | 0 | 1 | 2 | CS[C@@H](c1ccccc1[S@@](=O)c1ccc(C)cc1)[C@](C)(N[S@@](=O)c1ccc(C)cc1)c1ccccc1 |
| p8_targeted_hard | 50068534 | 726.7700 | 3.3400 | 4.3000 | 0.9600 | 12 | 4 | 0 | 0 | 6 | 6 | COc1cc(NC(=O)C(C)NC(=O)C2CCN(CC(=O)Nc3cc(C(=O)Nc4cccc(F)c4)ccc3C)CC2)ccc1NC(=O)c1cccc(F)c1 |
| ood1000 | 52916722 | 279.3700 | 6.4100 | 5.5000 | 0.9100 | 2 | 0 | 0 | 0 | 1 | 3 | CC1(C)O[C@H]2CC(/C=C/C(=O)N3CCCCC3)C[C@H]2O1 |
| p8_targeted_hard | 58373895 | 765.2500 | 2.9500 | 3.7800 | 0.8300 | 15 | 2 | 2 | 1 | 6 | 10 | COc1c(O)ccc(C(=O)NCC[N+]2(CC3=C(C(=O)[O-])N4C(=O)C(CC(=O)/C(=N\OC(C)(C)C(=O)O)c5csc(N)n5)C4SC3)CCCC2)c1Cl |
| ood1000 | 53907143 | 236.2600 | 4.9600 | 5.7500 | 0.7800 | 7 | 1 | 0 | 0 | 0 | 4 | CCC(C(=O)O)C(=O)COCc1ccccc1 |
| p8_targeted_hard | 24526198 | 460.5900 | 4.2300 | 3.4500 | 0.7800 | 8 | 4 | 3 | 0 | 4 | 3 | Cc1cc2nnc(SCC(=O)c3ccc(CCNS(C)(=O)=O)s3)n2c2ccccc12 |
| p8_targeted_hard | 89190026 | 753.0300 | 2.4500 | 3.1900 | 0.7300 | 8 | 5 | 0 | 0 | 2 | 0 | CC1(C)C2=CC(N(c3ccc(/C=C/c4ccc5c(c4)C(C)(C)c4cc(N(C6=CC=CCC6)c6ccccc6)ccc4-5)cc3)C3C=CC=CC3)CC=C2c2ccccc21 |
| ood1000 | 90836517 | 489.4300 | 3.2100 | 3.9000 | 0.6900 | 3 | 1 | 0 | 0 | 3 | 10 | CC1c2ccc([N+](=O)[O-])c(O)c2C(=O)C2C(=O)C3(O)C(=O)C(C(N)=O)C(=O)[C@@H](N(C)C)C3C(O)C21 |
| ood1000 | 61070951 | 295.4200 | 5.9200 | 6.6100 | 0.6800 | 6 | 0 | 0 | 0 | 1 | 3 | CCC1CCCCC1NC(=O)CC1(CC(=O)O)CCCC1 |
| p8_targeted_hard | 9897151 | 788.0200 | 5.3000 | 4.6300 | 0.6600 | 6 | 0 | 0 | 0 | 1 | 11 | C=CC[C@@H]1/C=C(\C)C[C@H](C)C[C@H](C)[C@H]2O[C@@](O)(C(=O)C(=O)N3CCCC[C@H]3C(=O)O[C@H](/C(C)=C/C3CC[C@@H](O)[C@H](OC)C3)[C@H](C)[C@@H](O)CC1=O)[C@H](C)C[C@@H]2OC |

The worst rows are enriched for narrow-gap, very large, flexible, or otherwise
structurally difficult molecules. They are not exclusively large molecules, so
the actionable diagnosis is the intersection of low Gap, size/flexibility, and
geometry sensitivity rather than molecular weight alone.

## Conclusion

Expansion500k is a real improvement over replacement300k in the exact bins where
the model is still weakest. The remaining error is concentrated in a small tail,
especially Gap <3 eV and MW>800/flexible structures. That tail overlaps two known
limits:

1. narrow-gap / charge-transfer chemistry, where B3LYP labels themselves are the
   method ceiling and GW Delta-learning is the right next accuracy lever;
2. very large flexible molecules, where ETKDG geometry quality can become a
   separate edge case.

So the conservative decision is: keep v3 as the B3LYP surrogate default, stop
head-swap/longer-B3LYP-training loops by default, and move the main work to
Phase 9/10 re-validation against v3. Another B3LYP targeted top-up is not
impossible, but it is now lower ROI than GW Delta-learning.
