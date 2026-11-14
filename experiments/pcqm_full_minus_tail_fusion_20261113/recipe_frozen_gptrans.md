# GPT-T member recipe, frozen 2026-11-14 before the run

Frozen by the 2026-11-14 desktop agent, before any GPT-T minus-tail result exists.
Authority: [protocol.md](protocol.md) ("GPT-T B1024 ... recipe frozen separately before its run").
No tail_holdout_v1 or official valid/test number was used to choose any value below.

## Values

| Item | Frozen value | Source |
|---|---|---|
| Model | GPT-T BondLocal + PairNorm (`gptrans_t`, mode `degree_bond_local_pair_update_norm_ema999`) | runner `make_recipe`, same as the 500K controls |
| Rows | [0, 3,250,000), `--max-shards 65` (shards 0000-0064) | protocol |
| Init | `gptrans_init.pt` seed 42, file sha256 `27590c1a...c3320`, tensor-state sha `a65034da...d1b` | runner default; same file as every seed-42 GPT-T run |
| Batch | 1024 (3,173 steps/pass, 190,380 steps, drop_last) | protocol (B1024) |
| Optimizer | AdamW, weight decay 0.05, grad clip 1.0 | runner recipe |
| LR | 2.8284271e-3 = 1e-3 x sqrt(1024/128) | sqrt-batch rule fixed in the 500K B1024 control protocol (61aad701) before its results |
| Schedule | 60 passes, 4 warmup passes, cosine to 1e-6 | runner recipe |
| EMA | 0.999^8 = 0.992027944 per step (erratum 3763f029) | same control |
| Endpoint | pass-60 EMA (`ema_model.pt`, selection `final-ema`); no snapshot choice | protocol |
| Precision | TF32 matmul, no compile | runner fast mode, same control |
| Normalisation | mean 5.69484785548952, std 1.1609073178422544 eV (`MINUS_TAIL_STATS`, rows 0:3.25M) | runner constant |
| Runner | `platforms/colab/pcqm_k1_b1024_production/train_full_b1024.py`, sha256 `120d5571593c56647afe8f7c6f6060afeb7781ff8aea101fb9454fdac0ee5b37` (repo = Drive `train_full_b1024_minus_tail.py`) | commit 5e6f12c7 |
| Monitoring | per-pass probe (20,480 in-train rows, seed 12345), snapshot every 3 passes, `sessions.jsonl`, `run_provenance.json` | runner |

## Differences from earlier GPT-T records

- The 2026-11-10 package in this branch (`pcqm_gptrans_full_minus_tail_20261110`) froze batch 128, fp32, lr 1e-3. The
  protocol names B1024, and 60 passes at batch 128 are 1.52M steps, not affordable. The protocol wins; the old package is
  superseded and unused. Its `training_recipe_gptrans_pairnorm.json` is not an input to this run.
- Evidence the recipe is weaker than batch 128 at 500K: GPT-T B1024 500K control was REJECTED under its own rule (tail
  111.315 meV mean vs batch-128 anchor 107.94; `experiments/pcqm_gptrans_b1024_precision_500k_20261112`). It is still the
  recipe the protocol fixes. Read the GPT-T member and the fusion with that gap in mind; do not re-tune after tail scores.
- Hardware/runtime: IMS A100 with torch 2.7.1 (recipe allows 2.7.1 and 2.11.0; the 500K controls and the K1 member ran torch
  2.11.0). `platforms/ims/gptrans_minus_tail/run_gptrans.py` sets only `recipe.torch_version` to the installed build and
  otherwise runs the registered runner unchanged. Runtime is recorded per session in `sessions.jsonl`. This is a confound
  against the 500K controls by design; it is not a tuning choice.

## Duplicate-identity note

A Colab L4 run of the same identity exists (`runs/gptrans-b1024-minus-tail-20261114`, 3 passes, last write 2026-11-14 22:58
JST, about 1,530 s/pass, so about 25 h for 60 passes). It was not touched. The IMS run is a separate output; only one of
the two may be scored as the GPT-T member, chosen by completion, never by tail score.
