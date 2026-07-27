# Phase 8 v2 Error Mode Analysis

This is a model-selection audit, not a new training run.

## Decision-Relevant Summary

- Common eval avg MAE: P7 0.14529 -> P8 0.12839 (-0.01690).
- Common eval Gap MAE: P7 0.17930 -> P8 0.15610 (-0.02320).
- PCQM proxy Gap MAE: P7 0.25444 -> P8 0.24645 (-0.00798).
- Main interpretation: replacement300k is a v2 B3LYP-base upgrade for low-coverage chemistry, while high-similarity chemistry is essentially tied.

## Common Eval By Slice

| eval_set | n | p7_avg_mae | p8_avg_mae | delta_avg_mae | p7_gap_mae | p8_gap_mae | delta_gap_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ood1000 | 999 | 0.12431 | 0.12144 | -0.00287 | 0.14881 | 0.14479 | -0.00402 |
| p8_targeted_hard | 978 | 0.16671 | 0.13548 | -0.03123 | 0.21044 | 0.16765 | -0.04279 |

## PCQM Proxy By P7 Similarity

| sim_bin | n | p7_gap_mae | p8_gap_mae | delta_gap_mae |
| --- | --- | --- | --- | --- |
| [0.0, 0.3) | 182 | 0.52733 | 0.49857 | -0.02876 |
| [0.3, 0.4) | 581 | 0.31846 | 0.29762 | -0.02084 |
| [0.4, 0.5) | 945 | 0.22933 | 0.22353 | -0.00581 |
| [0.5, 0.6) | 746 | 0.21355 | 0.21055 | -0.00300 |
| [0.6, 1.01) | 534 | 0.19331 | 0.19560 | 0.00229 |

## Remaining Worst Common-Eval Molecules

| eval_set | cid | gap | p8_avg_abs | p8_gap_abs | delta_avg_abs | flags | scaffold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| p8_targeted_hard | 25143081 | 1.57826 | 2.34067 | 3.50408 | -0.41159 | flexible; narrow_gap | O=C1CC2C(Cc3ccoc3)CCC34OC5OC67CC(CC6C(O1)C23O5)CC74 |
| p8_targeted_hard | 11263288 | 2.57692 | 1.28238 | 1.92465 | -0.05396 | flexible; large_conjugated; narrow_gap | O=C(Nc1ccccc1C(=NCCc1ccccc1CCN=C(c1ccccc1)c1ccccc1NC(=O)C1CCCN1Cc1ccccc1)c1ccccc |
| p8_targeted_hard | 57272463 | 2.70209 | 1.09289 | 0.60511 | -0.08269 | narrow_gap; sulfur | C1=CS2(CCNCC2)c2cncnc21 |
| p8_targeted_hard | 57228085 | 2.47351 | 0.93001 | 1.39296 | 0.07527 | narrow_gap; sulfur; chlorinated | C1=CSS(c2ccccc2)=N1 |
| ood1000 | 55542072 | 3.58102 | 0.82351 | 1.23678 | 0.01266 | flexible; large_conjugated; sulfur; fluorinated | O=C(CNc1ccccc1)Nc1ccccc1 |
| p8_targeted_hard | 25193465 | 3.53204 | 0.78621 | 1.13192 | -0.03001 | flexible; large_conjugated; sulfur | O=S(NC(Cc1ccccc1S(=O)c1ccccc1)c1ccccc1)c1ccccc1 |
| p8_targeted_hard | 57417452 | 3.06400 | 0.74893 | 1.11924 | -0.15590 | ordinary | O=C1CCCC=CC(=O)OC2CC(CCCC=CC=CCCCCC(=O)CCC1)OC1(C2)OCCCC1=O |
| p8_targeted_hard | 19826067 | 2.81366 | 0.72805 | 1.07758 | -0.14585 | salt_or_multifragment; flexible; large_conjugated; narrow_gap | c1ccc(C(c2ccccc2)N2CCN(CCCc3noc4ccccc34)CC2)cc1 |
| p8_targeted_hard | 508626 | 2.95788 | 0.72410 | 1.09175 | -0.19357 | flexible; narrow_gap; sulfur | O=C(CNc1ccccc1)NCCCCCNS(=O)(=O)c1ccccc1 |
| p8_targeted_hard | 50068534 | 3.34428 | 0.67642 | 1.01843 | 0.06820 | flexible; large_conjugated; fluorinated | O=C(CNC(=O)C1CCN(CC(=O)Nc2cccc(C(=O)Nc3ccccc3)c2)CC1)Nc1ccc(NC(=O)c2ccccc2)cc1 |
| p8_targeted_hard | 118707438 | 1.23268 | 0.60494 | 0.92035 | -0.13613 | large_conjugated; narrow_gap | O=C1CCC(=O)C2=C1C(=O)c1cccc(-c3cccc4c3C(=O)c3cc(-c5cccc6cc7c(cc56)CCCC7)ccc3C4=O |
| p8_targeted_hard | 87253780 | 4.09803 | 0.60043 | 0.89100 | 0.19565 | salt_or_multifragment; flexible; large_conjugated; fluorinated | O=C(NC1CCCC1N1CCCC1)c1ccccc1.O=C(NC1CCCC1N1CCCC1)c1ccccc1C1CC1 |
| p8_targeted_hard | 58373895 | 2.94971 | 0.59457 | 0.88307 | -0.25894 | flexible; large_conjugated; narrow_gap; sulfur; chlorinated | N=C(C(=O)CC1C(=O)N2C=C(C[N+]3(CCNC(=O)c4ccccc4)CCCC3)CSC12)c1cscn1 |
| p8_targeted_hard | 86736230 | 3.94837 | 0.57508 | 0.86194 | -0.01346 | flexible; sulfur | O=C(CNS(=O)(=O)c1cccc(C2CCCCC2)c1)N1CCCCC1 |
| p8_targeted_hard | 86685855 | 2.96604 | 0.56907 | 0.85123 | 0.07517 | flexible; large_conjugated; narrow_gap; fluorinated | O=C(CC(=O)NC(CC(=O)c1ccccc1)c1ccccc1)Nc1ccccc1 |
| p8_targeted_hard | 56960998 | 2.11432 | 0.54647 | 0.77175 | -0.34524 | flexible; large_conjugated; narrow_gap | c1ccc2[nH+]c3ccccc3cc2c1 |
| p8_targeted_hard | 86735774 | 2.74563 | 0.54504 | 0.68317 | -0.01874 | salt_or_multifragment; flexible; large_conjugated; narrow_gap | c1ccc(OCCN2CCCC2)cc1 |
| p8_targeted_hard | 24526198 | 4.23137 | 0.52753 | 0.79547 | -0.03878 | flexible; large_conjugated; sulfur | O=C(CSc1nnc2ccc3ccccc3n12)c1cccs1 |
| p8_targeted_hard | 25142765 | 5.06676 | 0.52238 | 0.79126 | -0.10675 | flexible; large_conjugated |  |
| p8_targeted_hard | 57072528 | 3.12387 | 0.51314 | 0.77345 | 0.01918 | flexible; large_conjugated | O=C(CCCCCc1ccccc1)c1ccccc1C(=O)Nc1ccc[nH]1 |
| ood1000 | 89495903 | 4.97424 | 0.51046 | 0.53387 | -0.00734 | flexible; sulfur |  |
| ood1000 | 94655998 | 4.65587 | 0.49516 | 0.74237 | 0.03550 | flexible; large_conjugated | O=C(NCCc1ccccc1)NCc1ccccc1 |
| ood1000 | 52916722 | 6.40828 | 0.49035 | 0.69984 | -0.07314 | wide_gap | O=C(C=CC1CC2OCOC2C1)N1CCCCC1 |
| ood1000 | 9857128 | 2.63678 | 0.48777 | 0.72802 | -0.03416 | narrow_gap; fluorinated | C1=NN=CC1N=Nc1ccccc1 |
| ood1000 | 53907143 | 4.96336 | 0.48423 | 0.72856 | -0.01064 | ordinary | c1ccccc1 |
| p8_targeted_hard | 9897151 | 5.29534 | 0.48395 | 0.73153 | 0.05335 | ordinary | O=C1CC=CCCCCC2CCCC(O2)C(=O)C(=O)N2CCCCC2C(=O)OC(C=CC2CCCCC2)CCC1 |
| p8_targeted_hard | 9896655 | 4.23409 | 0.47776 | 0.71969 | -0.03730 | flexible; large_conjugated; sulfur; chlorinated | O=C(NC(CSSCC(NC(=O)c1ccccc1)C(=O)NCCc1ccccc1)C(=O)NCCc1ccccc1)c1ccccc1 |
| p8_targeted_hard | 89190026 | 2.45447 | 0.47506 | 0.71580 | -0.07672 | flexible; large_conjugated; narrow_gap | C1=CCCC(N(c2ccccc2)c2ccc3c(c2)Cc2cc(C=Cc4ccc(N(C5C=CC=CC5)C5C=C6Cc7ccccc7C6=CC5) |
| ood1000 | 87801326 | 3.05584 | 0.47376 | 0.71399 | -0.01461 | flexible; sulfur; chlorinated | O=C(CCCCN1CCCCC1)c1cccc(OS(=O)(=O)c2ccccc2)c1 |
| p8_targeted_hard | 10995230 | 3.39054 | 0.46497 | 0.70054 | -0.05676 | flexible; sulfur | O=C(C1CC(SC2=CN3C(=O)CC3C2)CN1)N1CCCC1 |

## Remaining Worst PCQM Proxy Molecules

| pcqm_idx | gap_true | p7_sim | p8_gap_abs | delta_gap_abs | flags | scaffold |
| --- | --- | --- | --- | --- | --- | --- |
| 3553266 | 3.49122 | 0.55556 | 3.51369 | -0.01103 | radical_or_open_shell | [C]1CCC(CCCN2CCCCC2)CC1 |
| 3389082 | 3.65993 | 0.46341 | 3.51087 | -0.08260 | radical_or_open_shell | [CH]1[CH][CH][C](OC2CCCCO2)[CH][CH]1 |
| 3718398 | 2.40549 | 0.29412 | 3.23422 | -0.07722 | radical_or_open_shell; narrow_gap | c1ccccc1 |
| 3412547 | 2.03813 | 0.33898 | 3.20113 | -0.07404 | radical_or_open_shell; narrow_gap | [c]1ccc2[nH]c([CH]Cc3ccccc3)nc2c1 |
| 3418724 | 2.91706 | 0.42553 | 3.04952 | -0.03358 | radical_or_open_shell; narrow_gap | c1cn[nH]c1 |
| 3381494 | 2.13065 | 0.22727 | 3.04590 | 0.01728 | radical_or_open_shell; narrow_gap | [C]1C=CC2(C=C1)CN2OCC=NC1CC1 |
| 3400666 | 3.52932 | 0.24000 | 2.92988 | 0.03671 | radical_or_open_shell | C1CCCC1 |
| 3416953 | 3.55109 | 0.36842 | 2.69767 | 0.09380 | radical_or_open_shell | c1ccccc1 |
| 3417530 | 2.89257 | 0.25862 | 2.64143 | 0.12831 | radical_or_open_shell; narrow_gap | O=C1N=C[C]CN1 |
| 3547923 | 2.37828 | 0.26531 | 2.55252 | -0.27022 | radical_or_open_shell; narrow_gap; fluorinated | [C]1[CH][CH]C=CC1 |
| 3524417 | 2.32929 | 0.21277 | 2.35125 | 0.01472 | radical_or_open_shell; narrow_gap; sulfur | O=C(C=C1SCCS1)CCc1[c][c]co1 |
| 3390042 | 2.97965 | 0.42222 | 2.30175 | -0.10330 | radical_or_open_shell; narrow_gap; sulfur | O=S1(=O)CCC(N[C]2C[C][CH]C=N2)C1 |
| 3520329 | 3.66537 | 0.38000 | 2.22851 | -0.07182 | radical_or_open_shell; sulfur | [C]1CCC(N=CCCc2cccs2)C1 |
| 3427000 | 5.17016 | 0.50000 | 2.10813 | 0.09822 | radical_or_open_shell; flexible |  |
| 3465008 | 2.47351 | 0.27273 | 2.04367 | -0.39540 | radical_or_open_shell; narrow_gap | O=C1[CH]c2ccccc2[N]1 |
| 3405285 | 3.73612 | 0.26190 | 2.00516 | -0.11043 | radical_or_open_shell; sulfur | [C]1C=CC2(CCC3SCCS3)[CH][C]12 |
| 3398841 | 3.36061 | 0.40426 | 1.90104 | -0.15766 | radical_or_open_shell | c1c[nH]cn1 |
| 3608800 | 1.58914 | 0.42308 | 1.89915 | -0.03628 | radical_or_open_shell; narrow_gap | O=C(CN1[CH]c2ccccc2[CH]C1)c1ccccc1 |
| 3411735 | 2.91978 | 0.32258 | 1.86962 | 0.03097 | radical_or_open_shell; narrow_gap | [CH]1O[C]2C=CC=CC12 |
| 3473715 | 2.71842 | 0.28889 | 1.80828 | 0.05898 | radical_or_open_shell; narrow_gap | c1ccc2c(c1)NCN2 |
| 3666935 | 4.07899 | 0.36667 | 1.66849 | -0.04901 | radical_or_open_shell | O=C1[CH][C]2CC=C3C4CCCC4CCC3C2CC1 |
| 3412316 | 4.14702 | 0.28000 | 1.65739 | -0.23707 | radical_or_open_shell; flexible | [C]1C=CC2[CH][C]12 |
| 3411385 | 4.93615 | 0.25000 | 1.64059 | -0.22722 | radical_or_open_shell; chlorinated | [CH]1C2OC12 |
| 3734827 | 3.19462 | 0.35417 | 1.60402 | -0.09135 | radical_or_open_shell | c1ccc([N][N]c2ccccc2)cc1 |
| 3423329 | 4.15246 | 0.28000 | 1.57126 | 0.10979 | radical_or_open_shell | O=C1NC[C]2N[CH]CCC21 |
| 3704774 | 4.33477 | 0.34043 | 1.52047 | -0.04554 | radical_or_open_shell | O=C1[CH][CH]C2CCCCC2C1 |
| 3414265 | 5.52119 | 0.41176 | 1.47727 | 0.00114 | radical_or_open_shell; flexible; wide_gap |  |
| 3554614 | 4.00279 | 0.33962 | 1.45414 | 0.05311 | radical_or_open_shell | [CH]1CNC(c2ccccc2)O[N]1 |
| 3411990 | 3.76606 | 0.57500 | 1.43265 | 0.01149 | radical_or_open_shell | [C]1C=CC2(CCCNCc3ccccn3)[CH][C]12 |
| 3624948 | 3.14564 | 0.33333 | 1.43261 | -0.01530 | radical_or_open_shell | N=C1C=C[C]([CH]c2ccccc2)c2ccccc21 |

## Top-100 Flag Counts

Common eval remaining worst:
- flexible: 58
- large_conjugated: 57
- sulfur: 35
- chlorinated: 23
- narrow_gap: 20
- fluorinated: 20
- salt_or_multifragment: 12
- wide_gap: 11
- ordinary: 5

PCQM proxy remaining worst:
- radical_or_open_shell: 79
- narrow_gap: 15
- sulfur: 15
- wide_gap: 15
- flexible: 7
- fluorinated: 6
- ordinary: 6
- chlorinated: 3
- large_conjugated: 3

Largest PCQM proxy improvements:
- radical_or_open_shell: 44
- sulfur: 29
- ordinary: 24
- wide_gap: 16
- narrow_gap: 9
- chlorinated: 9
- large_conjugated: 7
- fluorinated: 6
- flexible: 2

