# Single-Point Substitution ΔGap Zero-Training Diagnostic: Frozen Protocol

Frozen 2026-11-24 JST before any scoring-block label read or remote execution.
Desktop-owned Track B, branch `codex/exp/pcqm-delta-gap-mmp-diag-20261124`,
platform `kaggle2` (`kaseichou`, single T4 allocation, zero training).

RML check (`molgap.research_memory lookup/gaps` on `pairnorm`, `delta`, `mmp`,
`cliff`, `substitution`, `contrastive`, `scaffold`, `rewir` on 2026-11-24): zero
prior trajectories for matched-molecular-pair ΔGap recovery or rewiring
controls.

## 1. Question and Decision Link

On fixed pass-60 EMA (`EMA@60`) checkpoints of 500K GPTrans-T PairNorm
(`init_s43`, `init_s44`), does mispredicting how much HOMO-LUMO Gap changes under
a single-point substitution sharing the same core scaffold and attachment site
(`ΔGap`) contribute at least `3.0 meV` of whole-block absolute Gap MAE above a
covariate-balanced non-MMP rewiring control?

- Single-molecule inference remains unchanged if a training experiment is ever
  proposed later, and its decision metric remains whole-dataset absolute Gap MAE.
- Literature basis: DeepDelta (`PMC10605784`, Fig. 1, Methods, Table 2) for
  pairwise difference supervision; Activity-cliff awareness
  (`s41467-026-75713-2`, Fig. 2, Methods) for structurally paired contrast.
- This diagnostic consumes zero training slots and releases zero training. The
  2-slot / 4-arm budget on `kaggle2` stays untouched unless this diagnostic
  clears all gates and a separate training protocol is approved.

## 2. Frozen Models, Checkpoints, and Replay Checks

Both backbones come from the strictly verified 500K GPTrans-T PairNorm pair on
`kaggle2` (`kaseichou/molgap-gptrans-init-noise-500k-k2` / `-src`), trained on
rows `[0, 500000)` and selected on `[500000, 550000)`:

| Arm | Manifest SHA256 | `last_checkpoint.pt` (`ema`, pass 60 / step 234,360) SHA256 | `selected_model.pt` SHA256 | `development_predictions.pt` SHA256 | `training_contract.json` SHA256 | `canonical_trace.json` SHA256 | Init state SHA256 |
|---|---|---|---|---|---|---|---|
| `s43` | `9f5131f307df7ee373e066c36afa96f17c84767c8712ead844b8073ce77457ac` | `af709cc655cbcbd33ede294f09576dca78caad1385e91d61520850ceb4b9c52a` | `13369b0ac07b71177f8d4a7a4c29f98c4ad9b892ac65fbb6e93b97a1a24a06ba` | `818b0b041919f53e31667c201ad926d56bf7d8e59c352056c912a2d63e6f6a62` | `471ad2fb6785a1af128fbb5cccbb9a81c56c0b1c90064b692d82a098285bbc28` | `722f43effa1ffe856ba2ce72b81e5be06f7231703a4c92247011f9954d1ff4c5` | `2ccd1eb79de4b82fffb58522b9e2325026d36f04f398be0d49d3eea55ee05f3c` |
| `s44` | `2a439ae42a10179897ebf51d11a8a6db03d45029647c15781ec84af86e7a0f0a` | `170b62f8123c4d872cc3fb50189e92e996db27743261982854420ca2faffe511` | `a680e311a04548e9440d4b68866501d6dd8c3b49e96c86315699621bd0457ceb` | `fc5a9f5e11c303a8c5c1d1603f75b8ba853c23755db8e48538c52593e193a29e` | `594da69200aac0818af0fb2f4e9fb7b3938b5b5e48a235cb485f826373e4e7b1` | `14792cc98fea96eae094f7d968680196e0ce336f603ea94d2482ed117a6228d1` | `2415cdbfea332db158086b2ee457ce95bcb90a2e3f87c04564146ba12a2c3573` |

Pre-inference verification (deterministic FP32 without TF32, `batch_size=128`,
mounted dev shard `kaseichou/pcqm4mv2-ogb-fixed-500k-scnet-v1/train/train_shard_0010.pt`,
`1a37dc5d458396b590044d978526822e6d1cb35df223de913a837dc7277d64c1`):
1. `selected_model.pt` matches the first 128 saved `development_predictions.pt`
   predictions within `<= 1e-4 eV`.
2. `last_checkpoint.pt["ema"]` (`EMA@60`) reproduces `canonical_trace.json`
   pass-60 `ema_dev_metric` (`0.101540938 eV` for `s43`, `0.101365402 eV` for
   `s44`) within `<= 1e-4 eV`.

## 3. Scoring Block Eligibility and Contamination Audit

Existing saved predictions fail strict eligibility (all `sel` files except C3's
`dev` `[500k,550k)` and `unseen` `[3100k,3150k)` were already scored/selected or
probed; `0 / 2,332,374` eligible untouched rows have saved `EMA@60`
predictions). Formal effect evaluation therefore runs one deterministic `EMA@60`
forward pass on **contiguous Shards `58–61` (`[2,900,000, 3,100,000)`) minus
`spread-fit`'s `fit_idx`**:

- `fit_idx`: exact deterministic 120,000-row draw (`np.random.default_rng(20261123)`
  without replacement from `[600000, 1850000) U [1900000, 3150000)`), removing
  `2,369 + 2,407 + 2,415 + 2,359 = 9,550` rows and leaving **`N = 190,450`**
  completely untouched ordinary rows (`47,631 + 47,593 + 47,585 + 47,641`).
- Zero overlap with `[0, 550000)` (train/dev), H1 (`550k..600k`), `spread-fit`
  (`fit_idx`, shard 37, shard 63), C3 (`shard 62`), `block-transfer`/`C2`
  (`shard 64`), or reserved `tail_holdout_v1` (`>= 3,250,000`).
- Official validation, test-dev, test-challenge, common, OOD, P8-hard, and
  `tail_holdout_v1` are untouched (`pcqm4m-v2.zip` `628ae612...` is never read
  beyond row `3,099,999` in Kaggle mode).
- Topology shards (`pcqm_full_train_v1/manifest.json`
  `7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d`):
  - `train_shard_0058.pt`: `621e11151b10f775f1cd266df18b91525d75412bca505f3f663abaf1c3a25acb`
  - `train_shard_0059.pt`: `ba5cb07de08ff6d47371b7715b6e7d3d8f6fc668876d1b44921178006f374146`
  - `train_shard_0060.pt`: `32551f291e2d10dc482e8f82d3955eeee014ca41e3f5a0e94ff454f4417616ed`
  - `train_shard_0061.pt`: `35272cfdb0e338cd9956fccb1ba72b0bdb17b3d1c64cb88f164e71270d36da12`

## 4. Real Pairing, Rewiring Control, and Blind Discipline

Strict four-stage isolation inside `probe_delta_gap.py`:

1. **Stage 1 — Real Single-Point Substitution Pairing (zero `y`, zero `yhat`)**:
   - Fragment valid molecules with `rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=True, maxCutBonds=30)`.
   - Retain cores `core-[*:1]` with `core_ha >= 6`, substituent `1 <= r_ha <= 12`,
     `core_ha >= r_ha`, and `core_ha * 2 >= mol_ha`.
   - Within each core bucket, deduplicate by non-isomeric canonical SMILES
     (`Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)`) so
     stereoisomers/duplicates never pair; exclude mega-buckets `> 300` entries.
   - Construct forbidden 1-cut MMP adjacency `all_mmp_adj` from all retained
     buckets (`<= 300` entries) plus sequential mega-bucket membership IDs for
     buckets `> 300` entries, and run deterministic disjoint matching `E_real`
     (vertices sorted by `(deg, i)`, partner chosen by `(deg, j)`), so every
     molecule appears at most once (`|E_real| = M`, `|V_paired| = 2M`).
   - Seal `stage1_real_pairs.npz` (with series cluster labels and SHA256).
2. **Stage 2 — Deterministic `EMA@60` Inference (zero score `y`)**:
   - Replay-verify `s43` and `s44` on dev shard 10, then predict deterministically
     on `N = 190,450` graphs using topology fields only (no `.y` read in Stage 2).
   - Seal `stage2_predictions.npz` (`pred_s43_ema60`, `pred_s44_ema60`).
3. **Stage 3 — Matched Non-MMP Rewiring Control & Balance Gate (zero `yhat`)**:
   - Read CSV labels `y` for the `190,450` rows and assert float32 equality
     against the packed topology shards.
   - Function `rewire_and_gate(...)` receives **zero model predictions**.
   - On the exact same vertex set `V_paired` (`deg = 1`, `unmatched == 0`), forbid
     all edges in `all_mmp_adj`, shared mega-bucket (`> 300`) 1-cut MMP core
     memberships, and identical non-isomeric canonical SMILES so `mmp_leak == 0`
     covers all retained 1-cut MMPs.
   - Compute exact fp16 Morgan (`radius=2, fpSize=2048`) all-pairs Top-K
     (`K_global=64`, `K_scaf=32`) on GPU; sweep Phase-1 Lagrange multiplier
     `lambda in {0.0, 0.04, 0.08, 0.12, 0.18, 0.25}` (`score = Tanimoto + lambda * same_Murcko`)
     using descending greedy + 3 augmenting-path passes + 1-edge augmenting
     swap; run `600,000` neighbor-guided + random 2-opt swaps (seed `20261124`)
     jointly balancing `same_Murcko`, Morgan Tanimoto, `|dHA|`, and `|dGap|`
     (means, SDs, decile histograms).
   - **Balance Gate (fail-closed before Stage 4)**:
     - `unmatched == 0`, `mmp_leak == 0`, exact vertex multiset equality;
     - `|same_Murcko_real - same_Murcko_rew| <= 0.02` (`2.0 pp`);
     - `|SMD| <= 0.10` and `KS <= 0.10` for `Tanimoto`, `|dHA|`, and `|dGap|`.
     - Any failure halts Stage 4 and closes the diagnostic as `insufficient_evidence`.
4. **Stage 4 — Single-Pass Scoring & Series-Clustered Bootstrap (runs iff Balance Gate passes)**:
   - For residual `r_i = yhat_i - y_i` (in meV), each pair `(u, v)` recovers
     `DeltaE_uv = |r_u| + |r_v| - |r_u + r_v|`. Whole-block recovery is
     `R = (1 / N) * sum_{(u,v) in E} DeltaE_uv` (unpaired rows contribute 0).
   - Exact vertex attribution `g_u = 0.5 * DeltaE_uv` preserves `sum_u g_u = N * R`.
   - **Series-clustered bootstrap (`B = 2,000`, seed `20261124`)**: hierarchical
     series clusters (`Murcko nrings >= 2` kept intact; single-ring/acyclic
     molecules partitioned by connected components of same-Murcko acyclic-R 1-cut
     MMPs; remaining unpaired singletons form singleton clusters). Resample
     clusters with replacement (`n_clusters` draws) using the paired ratio
     estimator `sum g / sum cluster_size` under identical draws for `R_real`,
     `R_rew`, and `diff = R_real - R_rew`.
   - Report raw Murcko-scaffold cluster bootstrap, 2-init equal-weight ensemble
     (`ema60`), paired-subset `ΔGap` MAE, and whole-block raw MAE descriptively.

## 5. Pre-Registered Verdict & Stopping Rules

| Verdict | Condition (evaluated on `s43` and `s44` individually) | Disposition |
|---|---|---|
| `insufficient_evidence` | Eligible block/predictions missing OR Stage 3 Balance Gate fails | Stop effect evaluation; no training |
| `STOP_NO_TRAIN` | Either `s43` or `s44` has `R_real` 95% CI upper `< 3.0 meV` | Close `ΔGap` auxiliary loss line; no training |
| `PASS_DIAGNOSTIC` | Both `s43` and `s44` have `R_real` point estimate `>= 3.0 meV` **AND** both have `(R_real - R_rew)` 95% CI lower `> 0.0` | Diagnostic passes; design separate training protocol (decision metric = whole-dataset Gap MAE) |
| `INCONCLUSIVE_DO_NOT_TRAIN` | All remaining cases | Do not advance to training (even if paired-subset `ΔGap` MAE drops strongly) |

## 6. Execution and Storage Policy

- Expected Kaggle T4 cost: ~15–25 min wall time, `< 0.5` T4 device-hours
  (zero training; 1 short diagnostic kernel).
- Local smoke check runs Stages 1 + 3 + synthetic-residual bootstrap strictly on
  already-consumed `Shards 63–64 \ fit_idx` (`95,245` rows, `< 3,250,000`, zero
  untouched rows and zero model predictions).
- Large row arrays (`predict`, `real_pairs`, `rewiring`) stay outside Git in
  `/kaggle/working/delta_gap_diag/` (retrieved to `D:/molgap-artifacts/diagnostics/`).
