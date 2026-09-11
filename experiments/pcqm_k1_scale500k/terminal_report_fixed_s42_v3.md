# K1 500K fixed-cache seed-42 GPU v3 terminal evidence

- Kernel: `kaseichou/molgap-pcqm-k1-scale500k-s42`, version `3`.
- Terminal state: `COMPLETE`.
- Retrieved evidence root: `platforms/_records/kaggle/training/pcqm_k1_scale500k_fixed_s42_v3`.
- Actual result root: `pcqm_k1_scale500k_s42`.
- Frozen no-inference acceptance: `experiments/pcqm_k1_scale500k/accept_result.py` completed with `accepted=true`.

## Frozen identity and contract

- Source commit: `36215d9539acdd75542608637ec1e2db5341d3ff`.
- Fixed dataset: `kaseichou/pcqm4mv2-ogb-fixed-500k-scnet-v1`.
- Cache SHA-256: `630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`.
- Fixed geometry aggregate SHA-256: `30b57ac10ddcd1decb7729b299b9b92fbfdf0fe7de15700b40bd488cd3a9ac4d`.
- SCNet aggregate SHA-256: `676a506c808402bc16a4437cc02286168239a8dd5de4451e992131eddb4f1b20`.
- Shared contract: seed 42, FP32, Tesla T4, physical batch 128, one device, 40 epochs, fused AdamW `lr=4e-4`, `weight_decay=1e-5`, cosine schedule to `1e-6`, loader workers 2.
- Frozen acceptance verified the 50K development role alignment, model/prediction/checkpoint hashes, paired bootstrap recomputation, and sealed flags. `model_inference_executed=false`, `official_validation_role_read=false`, `test_dev_role_read=false`, `shadow_labels_read=false`.

## Paired training results

| arm | best epoch | validation Gap MAE (eV) | mean epoch time (s) | derived 500K graphs/s | peak memory (MiB) | reserve | parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full_gps` | 35 | 0.1146809458732605 | 454.0458193907501 | 1101.210447595162 | 534.03564453125 | 0.9641867733258728 | 4,771,073 |
| `neural_atom_k1` | 38 | 0.10485909879207611 | 394.2719251265501 | 1268.1602927713257 | 436.6923828125 | 0.9707147576146228 | 3,658,817 |

The throughput column is the mechanical `500000 / mean_epoch_seconds` derivation; no separate throughput field was published. Both arms completed 40 epochs.

- Paired gain (`full_gps - neural_atom_k1`): `0.009821847081184387 eV`.
- Paired bootstrap 95% CI for (`K1 - full_gps`): `[-0.010503999279379844, -0.009154959342956544] eV`.
- Minimum gain gate: `0.001 eV`; recorded `scale_gate_passed=true`.

This report records the frozen acceptance outputs only; scientific coordination and any successor decision remain with the coordinator. No local model execution or successor submission was performed by this monitor.

## Retrieved artifact hashes

| artifact | bytes | SHA-256 |
|---|---:|---|
| `molgap-pcqm-k1-scale500k-s42.log` | 13307 | `13D73EFF4F9FF5C4C694C2BFB897BB694E9C85DD52DD0015D99B5915F441406B` |
| `pcqm_k1_scale500k_s42/acceptance.json` | 1001 | `9CE944B2F3F5133E64D28CA5B8FCB8C10E0AFDFC82FACA33C47CD3FCBA06A6CC` |
| `pcqm_k1_scale500k_s42/completion_manifest.json` | 6274 | `9FADE8BB8B0396641AB11048E7549539F2D7C18C8EDD314934FDC9E82B9487F0` |
| `pcqm_k1_scale500k_s42/metrics.json` | 5123 | `05E59AA2214D8696DA159A0D6F9BA88C2CEA7AFCE91334D3EC21E3C8020A5A1A` |
| `pcqm_k1_scale500k_s42/full_gps/best_model.pt` | 19265612 | `241FBF3BC3048A68F5B87401225B72852024347C2D2DFCBDE4141D625DEF2073` |
| `pcqm_k1_scale500k_s42/full_gps/best_validation_payload.pt` | 802389 | `05B4CDBA7743B4FE2D298D295A84805EE2E17F4B4E5FE227549AD3261C590BC6` |
| `pcqm_k1_scale500k_s42/full_gps/last_checkpoint.pt` | 57714466 | `B777F51CBCE4F592C83B2DE3223F4C686CF9140E03BFCE120B49C98E90C0AA82` |
| `pcqm_k1_scale500k_s42/neural_atom_k1/best_model.pt` | 14799508 | `D4DB17256EAE15500A1D1B59B270DE2173C160899D406BC28F7F643E7CD2321D` |
| `pcqm_k1_scale500k_s42/neural_atom_k1/best_validation_payload.pt` | 802389 | `ECD27EE2CA521708CEA067878CE3802CCAA7CB336E33C5FC5A5529B6958707E7` |
| `pcqm_k1_scale500k_s42/neural_atom_k1/last_checkpoint.pt` | 44352765 | `1B943D71B8165A5F585CE39482B2C3340B744786D082B6889FFAC0796B63FF6F` |
