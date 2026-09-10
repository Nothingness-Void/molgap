# PCQM K1 shadow cache v2 terminal evidence

- Kernel: `kaseichou/molgap-pcqm-k1-shadow-cache`, version `2`.
- Final Kaggle2 status: `COMPLETE`.
- Read-only status/output commands used `C:\Users\Adminn\Documents\molgap\.venv\Scripts\kaggle.exe` with `KAGGLE_CONFIG_DIR=C:\Users\Adminn\Desktop`.
- Retrieved evidence root: `platforms/_records/kaggle/training/pcqm_k1_shadow_cache_v2`.
- Frozen acceptance command: `.venv\Scripts\python.exe experiments/pcqm_k1_shadow/accept_cache.py`.
- Acceptance result: `accepted=true` (no model inference and no shadow-label read).

## Frozen identity and sealed-role contract

- Launch manifest: `experiments/pcqm_k1_shadow/results/cache_launch_v2.json`.
- Launch manifest SHA-256: `E50F361F6351C2142D3F4BFB0D607BF402A8A38A6F0FFE6FD075A4E7C4522392`.
- Source commit: `aa81281edf15fc1516f036ef9850a39b7ee669d0`.
- Frozen candidate: `neural_atom_k1`; candidate selected before shadow labels.
- Parent graph-cache aggregate SHA-256: `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`.
- Parent geometry-cache aggregate SHA-256: `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
- CSV columns read: `idx`, `smiles` only.
- Cache: 10,000 graphs, 10 shards of 1,000, atom feature dim 9, bond feature dim 3, RWSE dim 16.
- Failure count: `0`; reserve rows consumed: `0`; remote elapsed time: `139.69651150100003 s`; RDKit: `2026.3.6`.
- Flags: `shadow_labels_read=false`, `official_validation_role_read=false`, `test_dev_role_read=false`, `model_inference_executed=false`, `gpu_used=false`.

## Acceptance values

- Acceptance file: `pcqm_k1_shadow_cache/acceptance.json`.
- Acceptance SHA-256: `4A877A195B3C5F500963AC3C15AEEE0F2FD523316ECA39E53674EA67C50A0771`.
- Cache aggregate SHA-256: `4a9e4361247f497f5cd9911a69e69eddb1e0fecea6e33f318ada48aa594f6a44`.
- Effective shadow-index SHA-256: `f68f0223dccb1c0f80e76035c79efe08b0959e4e14d5b1f7ebb5153d6cbc27bd`.
- Split SHA-256: `47e15a03dd4d61b1f57bc1568f2113699d59e9e932c8c25d5435350eca4df4dd`.
- Failures SHA-256: `52aefa1e80d991c52b8fe85c498c0824a91de9afa4af428a0a076a063fc6503f`.

The frozen acceptance verified shard hashes, split/failure hashes, aggregate identity, row order/disjointness, graph structure and finite RWSE values. No shadow Gap labels, official validation/test-dev roles, model execution, audit GPU, or successor task was accessed or submitted by this monitor.

## Retrieved artifact hashes

| artifact | bytes | SHA-256 |
|---|---:|---|
| `molgap-pcqm-k1-shadow-cache.log` | 2669 | `2C6973F3F049ABCE80BA22D6D84471F6D7E46F3CC0EFCF27BEF9A03E25C01385` |
| `pcqm_k1_shadow_cache/acceptance.json` | 569 | `4A877A195B3C5F500963AC3C15AEEE0F2FD523316ECA39E53674EA67C50A0771` |
| `pcqm_k1_shadow_cache/failures.json` | 70 | `52AEFA1E80D991C52B8FE85C498C0824A91DE9AFA4AF428A0A076A063FC6503F` |
| `pcqm_k1_shadow_cache/manifest.json` | 3175 | `DD4F6866AC1689F9366660F74D8CAEB71B854C4974A7AD2DEFC14EDBA2C0AE07` |
| `pcqm_k1_shadow_cache/progress.json` | 128885 | `79F293A3294013619379EA7E2EBBE5178D816888D81142BB3F82A3A6F842A71A` |
| `pcqm_k1_shadow_cache/run_summary.json` | 417 | `35B34688F01E58E12920CCEF62EE935626CF167A10B8C56EEE7B2CA012E74B04` |
| `pcqm_k1_shadow_cache/shadow_part_000.pt` | 4452663 | `393AB4EA385D6195B2D367F4E74A603842669740686F7770AD20894E32BA3B60` |
| `pcqm_k1_shadow_cache/shadow_part_001.pt` | 4574071 | `FDB018F2A44A239715BA333309E125AAEA0BF9B26DD12296FA279986F1998E38` |
| `pcqm_k1_shadow_cache/shadow_part_002.pt` | 4676023 | `BE830C809F2CBD836417A210D615D5A993161A06340426083EA3270C93D267F4` |
| `pcqm_k1_shadow_cache/shadow_part_003.pt` | 4680055 | `D4170591C9AAF607331DBB4C2CC2433EB1AA7B001CCD1F85080C3376B4697610` |
| `pcqm_k1_shadow_cache/shadow_part_004.pt` | 4599799 | `25A34385BD5EB419D8422BA2883E83ADB1B8796A34B74722AF53485401F09C36` |
| `pcqm_k1_shadow_cache/shadow_part_005.pt` | 4297207 | `6DC0B4103F1E947F23EA054223416220C46AA777F09C332644F0E4CF3FB1FAEF` |
| `pcqm_k1_shadow_cache/shadow_part_006.pt` | 4532407 | `F5ECA4147FB11970636985D24B28D3B90913B7A115462B7FD7C26865E1D174D3` |
| `pcqm_k1_shadow_cache/shadow_part_007.pt` | 4448311 | `90BED2A79E584036A8210DDC9DDF496DBEBA8C2E5CC24F7319DAFD8989A7B899` |
| `pcqm_k1_shadow_cache/shadow_part_008.pt` | 4308727 | `D0FF1DC1A7D12000C5F6920B8FA99611F326EAB243E1740592F919AB2EC22730` |
| `pcqm_k1_shadow_cache/shadow_part_009.pt` | 4308663 | `78269F6E365B5E37397FA66B31C2F1BBD2063773FBB11F101D9194365739283B` |
| `pcqm_k1_shadow_cache/split.json` | 257251 | `47E15A03DD4D61B1F57BC1568F2113699D59E9E932C8C25D5435350ECA4DF4DD` |
