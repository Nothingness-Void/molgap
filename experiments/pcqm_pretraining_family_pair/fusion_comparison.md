# Retained fixed fusion comparison, 2026-10-01

The user requested the missing unpretrained blend. This supplements the existing
pretraining question; no training, checkpoint inference, weight fitting or
protected-role access was performed. The original terminal dispositions remain
INCONCLUSIVE pending strict V5 qualification.

The frozen local analysis plan is [fusion_comparison_plan.json](fusion_comparison_plan.json).
The machine-readable result is [fusion_comparison.json](fusion_comparison.json).
The thin [compare_fusions.py](compare_fusions.py) wrapper reuses the existing
`pcqm_gptrans_100k_transfer_control.analyze_pair.paired_metrics`, shared SHA256
and atomic JSON helpers. All four prediction-file hashes, finite values, exact
50,000 source indices 100000..149999 and identical targets passed. K1 canonical
row/target/prediction tensor hashes and GPTrans reference-model binding passed.
Both scratch single-model MAEs reproduce their historical records within 1e-7 eV.

| Retained endpoint | MAE (eV) |
|---|---:|
| Unpretrained K1-v4 | 0.141373641 |
| Unpretrained GPTrans Noisy Nodes + PairNorm, EMA | 0.147245052 |
| Unpretrained fixed 50:50 blend | 0.134480145 |
| Pretrained K1-v4 | 0.141176449 |
| Pretrained GPTrans Noisy Nodes + PairNorm, EMA | 0.146358246 |
| Pretrained fixed 50:50 blend | 0.134096618 |

Unpretrained fusion improves its K1 endpoint by **6.893496 meV**, paired-row
95% CI **[6.301273, 7.512456] meV**. Pretrained fusion improves its pretrained
K1 endpoint by **7.079831 meV**, CI **[6.499953, 7.720367] meV**.
The additional pretrained-blend improvement over the unpretrained blend is only
**0.383527 meV**, CI **[-0.317183, 1.057642] meV**; the pretrained blend improves
50.608% of individual rows. The interval spans zero and the point gain misses
the existing 3 meV nomination floor. It neither establishes a material extra
benefit nor proves that pretraining has no benefit.

The strongest supported observation is complementary prediction errors from
the two retained families: most of the blend benefit is present without
pretraining. There is no evidence here to prioritize this pretraining recipe
for scale-up. This is post-hoc, one-seed analysis of an already used development
role. Row bootstrap measures row sampling uncertainty, not training stochasticity.
Strict historical trace/runtime/downstream comparability and compute-efficiency
qualification remain unresolved; no model promotion or replay-ready claim follows.

The K1 reference payload was retained under the SCNet records directory with
filename `neural_atom_k1_v4.pt`. Its SHA256 exactly equals the canonical Kaggle
reference prediction manifest's `best_development_payload.pt` SHA256
`966ed31ba25e024aa82d8032e7ab6e2797402e5845a01888c8f3cfd4c084da91`.
The directory name is a storage location, not a new training-platform claim.
GPTrans predictions were retained by the accepted feature-denoising analysis;
its completion record and payload bind the original joint reference model
`c841cdee799daa7a874e0f112ce6dea2932fe0f640f434812bac15b83684b092`.
Their exact physical paths and immutable hashes are recorded in the plan/result.

The approximately 7e-10 eV difference from the earlier pretrained-blend report
is floating-point evaluation of an explicitly FP32 fixed blend. It does not
change any interpretation. Metrics and paired bootstrap are accumulated in
float64 by the reused analysis helper (1,000 replicates, seed42).
