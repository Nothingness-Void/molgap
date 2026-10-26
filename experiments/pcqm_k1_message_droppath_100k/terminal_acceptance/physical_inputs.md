# K1 Message DropPath 100K physical input evidence

Compiled 2026-10-27 JST from the retained submission, frozen source package,
fixed-subset manifest, and verified Kaggle v1 outputs.

## Exact launch and declared mounts

- Kernel: nothingnessvoid/molgap-k1-message-droppath-100k-s42, Kaggle kernel
  140239597, version 1.
- Submission record:
  experiments/pcqm_k1_message_droppath_100k/submission/record.json
  (exact account, kernel/version, source archive, package and Spec binding).
- Launch receipt:
  experiments/pcqm_k1_message_droppath_100k/submission/launch_receipt.json.
- Frozen Kaggle metadata:
  D:/molgap-artifacts/runs/pcqm-k1-message-droppath-100k-s42/v1/prepared/kernel/kernel-metadata.json,
  SHA256 5af97ca8812547cb7122b416c6574f9ee2596793efea53596be1874004899c8.
  Its datasetSources are exactly
  nothingnessvoid/k1-message-droppath-100k-src and
  nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1.
- Frozen entry script:
  D:/molgap-artifacts/runs/pcqm-k1-message-droppath-100k-s42/v1/prepared/kernel/run.py,
  SHA256 4fd34ffb8bb20bbf3fba1969d4e687e7695ad039a0c32911975b4650241f36aa.
  The entry verifies the frozen launch and source archive, then chooses the
  manifest roles from each frozen recipe and verifies only their shard paths.
- Source archive:
  D:/molgap-artifacts/runs/pcqm-k1-message-droppath-100k-s42/v1/prepared/package/source.tar.gz,
  SHA256 708f538ca5b5f5f44ab50ed307802282313770db8bcae0a1a9f26e0da58a95b3.
  In that archive, src/molgap/family_training_data.py has SHA256
  8f7539309cb86083104ae3ded83e6606a3a2bbd31b4c0cc6127806f798582f31.
  Its load_training_data reader selects only declared train and optional
  internal-development shards, and validates each decoded source_idx against
  the declared contiguous row range before using the rows.

## Frozen row membership and targets

The physical manifest at
D:/molgap-data/pcqm_fixed_100k_v1/manifest.json has SHA256
1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d,
matching both frozen arm recipes and output training contracts. The source
identity is OGB-LSC PCQM4Mv2, with no external data. The manifest has only:

| Role | Source indices | Rows | Membership SHA256 |
| --- | --- | ---: | --- |
| train | [0, 100000) | 100000 | baa5f49fbad78af4964d9ec7eaf2d6327b2d2ca1f4dcf54e2394dfff2e36d58e |
| development | [100000, 150000) | 50000 | 3270b30071692d712c468fa52fa247d37f0a8e4b5444dc7c913aadb8ca0832c4 |

The frozen baseline data declaration containing those role memberships has
SHA256 2ec53b73e69183aa1b51bdd01aa4180e87ecba34398510aff6a27cbc25c870be.
The shared target manifest
experiments/pcqm_k1_slot_width96/reference_binding/target_manifest.json has
SHA256 9c5b2f4a161cc3d6539cfa7bca89da37773276b201d775b1ab88300ac4c9de10.
Both output contracts bind 50,000 development rows, source_idx SHA256
3270b30071692d712c468fa52fa247d37f0a8e4b5444dc7c913aadb8ca0832c4, and target
SHA256 a1c0b711815225f0bccee905419d124174e1bf01c76877f5c661b063ee170137.
Their train and development usage identities match the frozen declaration.

## Protected-role evidence

The fixed-subset manifest lists train and development shards only; it sets
official_validation_role_read, test_dev_role_read, and
test_challenge_role_read to false, source_rows_included to false, and
external_data_used to false. The frozen reader requests train plus
development when the recipe enables development and does not select any
official-validation or test role. The declared Kaggle mounts include no
separate official-validation or test dataset. The hash-verified contracts,
traces, and saved development predictions bind both arms to this same
manifest and to the declared row identities.

The kernel metadata enables internet for package bootstrap. The untouched
claims here are scoped to the frozen, hash-bound readers and declared data
mounts for this run; no claim is made about data paths outside that execution
contract.

## Verified result artifacts

The retrieved per-arm output manifests are baseline
D:/molgap-artifacts/retrieved/pcqm-k1-message-droppath-100k-s42/v1/experiment/baseline/output_manifest.json
(SHA256 571f546e405699ad6f205f7e00907fdcbb475e05316ac33d500ca0ab6737bf23)
and message_droppath
D:/molgap-artifacts/retrieved/pcqm-k1-message-droppath-100k-s42/v1/experiment/message_droppath/output_manifest.json
(SHA256 eb1cf46e0aadd2d76e28702d1ab001b68b64a5a724c1e541f12cc9812135033d).
The manifests bind each training contract, canonical trace, and development
predictions used for the mechanical inspection and paired diagnosis.
