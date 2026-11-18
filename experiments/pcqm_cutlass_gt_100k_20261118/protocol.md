# CutlassGT-2026 at 100K: Protocol

2026-11-18 JST. Desktop-owned Track B pair (`nvoid912/molgap-cutlass-gt-100k-20261118-s42`).
- **Track**: Pure-2D Zero-4D-Tensor Fused Graph Transformer (`standalone_model=True`).
- Arms: `pair_update_norm` (same-job control from `gptrans.pt`) vs `cutlass_gt` (5-block sparse `BondLocal` + 8-head Cutlass SDPA with sign-free WIRE RoPE, FlashBias, ResFormer value residual, Gated Attention, and Pre-LN SwiGLU).

## Fixed Contract
- Split: `pcqm_fixed_100k_v1` (100K train, 50K internal development). Official validation and test roles remain sealed.
- Training: AdamW (`lr=1e-3`, `wd=0.05`, 4-pass warmup + 60-pass cosine), `batch_size=128`, FP32/no-TF32, EMA `0.999` + aux EMA `0.9999`, seed `42`, T4x2 one arm per device, `12600s` wall ceiling.
- Material gate: `cutlass_gt - pair_update_norm <= -3.0 meV` on selected EMA dev MAE and paired 2000-bootstrap 95% upper bound `< 0`.
