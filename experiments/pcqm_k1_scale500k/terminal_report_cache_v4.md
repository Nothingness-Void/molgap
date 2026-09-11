# K1 500K cache v4 terminal evidence

- Kernel: `kaseichou/molgap-pcqm-k1-scale500k-cache`, version `4`.
- Terminal state: `ERROR`; no GPU training was submitted.
- Retrieved evidence root: `platforms/_records/kaggle/training/pcqm_k1_scale500k_cache_v4`.
- The frozen `experiments/pcqm_k1_scale500k/accept_cache.py` acceptance was not run because the CPU cache job is incomplete and published no complete cache manifest/aggregate.

## Frozen roles and identity

- Cache worker source commit: `754eee2314baaa67e3e5d2b180f579008e201101`.
- Train role: source indices `0..499999`, exactly `500000` rows; SHA-256 `a9c8b2b698c67f30348c6edbccff00eb9e2c06b064ee4d1a11e607f9531a0f8e`.
- Development role: source indices `500000..549999`, exactly `50000` rows; SHA-256 `9ae885e5e74d82820d83758942eaf3e6ae2a7dcefbc1f0f4c174ce94c6786bb9`.
- SCNet reference cache aggregate SHA-256: `676a506c808402bc16a4437cc02286168239a8dd5de4451e992131eddb4f1b20`.
- Split SHA-256: `C6B3FC509AFE5FEC7B29B58104160365388214156E06157FD24B97339056BA7C`.

## Mechanical diagnosis and role counts

The worker built ten train shards (`50000` graphs total) and no development shard before failing at train slot/source row `51128`. RDKit rejected a five-valent Si SMILES (`Explicit valence for atom # 1 Si, 5, is greater than permitted`); `MolFromSmiles` returned `None`, OGB `smiles2graph` raised `AttributeError: 'NoneType' object has no attribute 'GetAtoms'`, and the cache builder wrapped it as `RuntimeError: SCNet-matched 500K/50K role contains an unparseable graph`.

The failure is mechanical and the partial cache is not eligible for GPU training or scientific comparison. Sealed flags in the retrieved split/progress are `official_validation_role_read=false`, `test_dev_role_read=false`, `shadow_labels_read=false`, `target_labels_read=false`, and `molecular_records_read=false`.

## Retrieved artifact hashes

| artifact | bytes | SHA-256 |
|---|---:|---|
| `molgap-pcqm-k1-scale500k-cache.log` | 4282 | `8F025ADD7C72F9F12855A86EC1AE3FFE53673184FEC9BAEF5387F9116890E25A` |
| `pcqm_k1_scale500k_cache/failures.json` | 255 | `1EE7BA949C08ECA46D31715AB9CAD4112F7CBD7511416D4B0A90FDE552DC1171` |
| `pcqm_k1_scale500k_cache/progress.json` | 3013 | `7F4E14C240393EE17A2E6F94E4EC0B9CA72555417D0FB37E63C0AF85F17DC434` |
| `pcqm_k1_scale500k_cache/split.json` | 6489682 | `C6B3FC509AFE5FEC7B29B58104160365388214156E06157FD24B97339056BA7C` |
| `pcqm_k1_scale500k_cache/train-0000.pt` | 25171705 | `57194E4685C3463DE8C37739AFBB48FAE05272319861D2C9C5B26BB673007D4D` |
| `pcqm_k1_scale500k_cache/train-0001.pt` | 24511353 | `7A68EE8451586FDFBF8BDE66091B9B511E63D4705C80640E44C3B6AC540DA88E` |
| `pcqm_k1_scale500k_cache/train-0002.pt` | 25662777 | `B702D15D5F62F1636A197171A225D08E77ED2972934ACC4E92D7B52B61E7971E` |
| `pcqm_k1_scale500k_cache/train-0003.pt` | 25044985 | `73322FCED0A489426B7D2D2942FF3B0456D77846AEC7C8700110AFFB575FFB4D` |
| `pcqm_k1_scale500k_cache/train-0004.pt` | 25461561 | `35DE098B588E0009F5EB096F72AE3563915E00752B7DAE67CEB7C9773555479F` |
| `pcqm_k1_scale500k_cache/train-0005.pt` | 24507705 | `D257EF1A0CD100F7AC9742E41B7E028DDB16D358AB5131476C100B3894E6397A` |
| `pcqm_k1_scale500k_cache/train-0006.pt` | 25192505 | `2E55878C36C53869AF904F75A4598DF88D5145CD457A91E97C5A6252ECDDFAF5` |
| `pcqm_k1_scale500k_cache/train-0007.pt` | 24102073 | `D9C9758082980D6AC8D8C82AE1751CAAC8676ED3AE00302E6BB7B23417343598` |
| `pcqm_k1_scale500k_cache/train-0008.pt` | 24949561 | `6FC4AD58AE76AE16DB87C549A2D0C6DD5A6828EAB530C5EB83336E3B31A5E826` |
| `pcqm_k1_scale500k_cache/train-0009.pt` | 24581817 | `E897BDB6DB65A70B82A77CD06467B80FF6C8728FAFED78565A3ADCBCE9EA6E6E` |
