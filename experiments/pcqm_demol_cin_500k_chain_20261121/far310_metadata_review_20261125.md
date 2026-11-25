# Far310 500K fusion metadata review

2026-11-25 JST. Disposition: exploratory observation, historical import pending the original Far310 scoring receipt. This review does not adjudicate the J3/J4/J5 training contract.

The [retained aggregate](D:/molgap-artifacts/retrieved/demol500k-chain-20261121/analysis/dev_and_farblock_eval.json) (SHA-256 `5e1b29326096437d7d9860318d8ee335435a3f7ec4f43dc1a18905c524890aa0`) reports 92.247 meV for 50/50 J3 DeMol-CIN + C500 GPTrans and 95.482 meV for 50/50 Kaggle GPTrans s43 + the same C500. For the fixed 60-pass comparison, the report gives a paired difference of -3.236 meV, with 95% interval [-3.679, -2.791]. These values describe the saved report for the stated 50K Far310 rows [3,100,000, 3,150,000).

| Member identity corroborated by training records | 60-pass last checkpoint SHA-256 | Expected Far310 prediction filename, present in the analysis directory |
|---|---|---|
| J3 `demol500k-chain-20261121/j3-demol-500k-s42` | `85a29c7ab3ca99eed3cb9e5605659ee4120662eff472a4f0916d9901db970cbf` | `j3_demol_500k_ep60_far310.pt` |
| C500 `gptdft-chain-20261121/c500-gptt-500k-s44` | `0d3f2506f4c6521de0a387b1a6ffeb99d86f53319ea38f2f479f1d04686d7595` | `c500_gptt_500k_ep60_far310.pt` |
| Kaggle2 `pcqm-gptrans-pairnorm-init-noise-500k-s43`, version 1, `init_s43` | `af709cc655cbcbd33ede294f09576dca78caad1385e91d61520850ceb4b9c52a` | `kag_s43_ep60_far310.pt` |

The [J3 retrieval receipt](D:/molgap-artifacts/retrieved/demol500k-chain-20261121/training/retrieval_partial_j1_j2_j3.json), [C500 retrieval receipt](D:/molgap-artifacts/retrieved/gptdft-chain-20261121/training/retrieval_legacy.json), and [Kaggle output manifest](D:/molgap-artifacts/retrieved/pcqm-gptrans-pairnorm-init-noise-500k-s43/v1/version1/init_s43/output_manifest.json) bind training outputs. They do not bind the Far310 exports. The J3 and C500 selected development metrics are consistent with the separate pass-40 and pass-52 entries in the aggregate. The training-side selection was development EMA, which does not establish the weight field loaded for the three pass-60 exports.

**Missing original scoring receipt.** Looked in the owning branches, both retrieval roots, and the Kaggle comparator output metadata. The retained material does not bind the three prediction exports to exact attempts, checkpoint SHA-256s and loaded EMA/live fields, scoring source/config, ordered row and target hashes, or paired-bootstrap settings. The [original role plan](role_plan.json) covers train/development and an offline tail convention. An applicable Far310 role authorization/use record for this scoring was not found. A report flag and matching filenames do not establish that attribution.

Stopped at the metadata gate. Prediction payloads and labels were not opened. No model inference, training, or remote action ran. Retain the report and its files as an exploratory observation. Reopen the historical import only if the original scoring receipt and contemporaneous role record are recovered.
