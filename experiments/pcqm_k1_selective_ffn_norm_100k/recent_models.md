# Primary-source normalization review

Review date: 2026-10-26. Literature/code review, not model qualification.

| Model | Verified placement and recipe | Primary source |
|---|---|---|
| GRIT (ICML 2023; historical pure-2D context) | PCQM explicitly enables post-residual node/pair BN after attention and node BN after FFN; disables LN. 16 layers, width256, BS256, AdamW LR2e-4/WD0, 150 epochs/10 warmup. | [Config](https://github.com/LiamMa/GRIT/blob/6c988ea600a606fbb49a2246c64a2d37396b3ab5/configs/GRIT/pcqm4mv2-GRIT-RRWP.yaml), [forward](https://github.com/LiamMa/GRIT/blob/6c988ea600a606fbb49a2246c64a2d37396b3ab5/grit/layer/grit_layer.py#L249) |
| GPTrans-T (IJCAI 2023; historical pure-2D context) | Node pre-LN before attention/FFN, embedding LN and readout LN; pair residual has no corresponding pair LN. Two learned channel-wise node residual gains initialized1.0. PCQM-T: 12 layers, node256/pair32, 300 epochs, fused AdamW/WD0.05, EMA0.9999; config notes actual LR1e-3. | [Block/readout](https://github.com/czczup/GPTrans/blob/c0be01f938570b6cb58e27f6e290bb4a78d1b5a4/models/gptrans.py#L195), [config](https://github.com/czczup/GPTrans/blob/c0be01f938570b6cb58e27f6e290bb4a78d1b5a4/configs/pcqm4mv2/gptrans_tiny_pcqm4mv2.yaml) |
| ESA (Nature Communications 2025) | Molecular QM9/DOCKSTRING commands request post-BN, but the released BN class actually uses BatchRenorm1d. LN alternative is feature-wise per edge token. Post paths use Admin residual modules. Molecular example: BS128, LR1e-4, WD1e-10, clip0.5, BF16. These are example recipes, not a verified optimal PCQM config. | [Molecular commands/caveats](https://github.com/davidbuterez/edge-set-attention/blob/861a0b12d445ae80cc69b3948af623b4331321d4/README.md), [normalizers](https://github.com/davidbuterez/edge-set-attention/blob/861a0b12d445ae80cc69b3948af623b4331321d4/utils/norm_layers.py), [residual placement](https://github.com/davidbuterez/edge-set-attention/blob/861a0b12d445ae80cc69b3948af623b4331321d4/esa/masked_layers.py#L301) |

## Recent architectures and leaderboard scope

GraphGPT (ICML2025) uses token pre-RMS before attention/MLP and final RMS in
its [released backbone](https://github.com/alibaba/graph-gpt/blob/main/src/models/graphgpt/utils_graphgpt.py#L135).
This is not graph-wise RMS; current main is not an immutable reconstruction of
every ranked submission. GPS++ (2022/2023) uses node post-residual LN and internal
node/edge/global MLP LN, with different edge/global residual treatment; its
[paper Appendix A.1](https://arxiv.org/html/2302.02947v2#A1.SS1) is historical context.

The [official OGB table](https://ogb.stanford.edu/docs/lsc/leaderboards/#pcqm4mv2)
ranks hidden test-dev MAE, separately listing validation and ensemble status;
it is not a pure-2D-from-scratch leaderboard. GraphGPT-L48 is explicitly marked
use3D (0.0709), distinct from GraphGPT MLM-tv0 (0.0802). GPS++'s winning recipe
uses 3D information/denoising. ESA reports official validation as test with400
fixed epochs and warns that released code was refactored; ESA/GRIT are absent
from that official table. No unverified2026 architecture is claimed here.

## Bounded K1 adaptation

`k1_local_residual_gain` transfers GPTrans's unit-initialized residual gains,
not its full architecture or normalization recipe. Each of nine original K1
local blocks reuses its exact conv, MLP, norm1/norm3 BN and
dropout objects/fields. Only the message and FFN residual updates gain192-channel
vectors initialized1, adding3456 scalar parameters in total. Existing checkpoint
keys remain unchanged; each block adds gamma1/gamma2. Substrate mutation is
reconstructed before strict candidate-checkpoint loading and optimizer creation.

This changes residual balance, not normalization scope or stochastic DropPath.
It does not repeat direct-LN, pre-RMS, GraphRMS, MsgNorm or BatchRenorm. The
targeted RML/canonical search found no K1 residual-gain trial; an older GPTrans
proposal did mention LayerScale. Benefit is hypothetical, not literature-proven
for K1. The user explicitly authorized local GPU100K/40epoch execution.
Synthetic test passes do not establish native qualification, scientific
acceptance, promotion or submission readiness.
