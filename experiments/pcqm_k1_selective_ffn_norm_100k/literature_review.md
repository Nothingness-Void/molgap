# K1 normalization review - 2026-10-26

## What K1 actually implements

K1 is a custom hybrid, not an exact reproduction of a published model:
OGB atom/bond encoders and RWSE16 feed nine persistent-edge ResGated/GPS
local blocks; a single-active-slot neural-atom-inspired mixer follows blocks
3, 6 and 9. Hidden width is 192 and slot width is 64.

The local message and FFN residual outputs each use BatchNorm (18 modules).
Persistent edge updates and slot processing already use LayerNorm. Thus K1
is a mixed-normalization model. Its PyG ResGatedGraphConv is not the original
GraphGPS CustomGatedGCN implementation.

Code owners: [factory](../../src/molgap/qm9_neural_atom.py),
[persistent edges](../../src/molgap/gps.py), and the component-level source
links in [recent models](recent_models.md).

## Primary-source findings

The implementation matrix and leaderboard caveats live in
[recent models](recent_models.md). Historical ancestry is distinct from recent
evidence: [original GraphGPS PCQM](https://github.com/rampasek/GraphGPS/blob/main/configs/GPS/pcqm4m-GPS%2BRWSE.yaml)
enables block BN; [Neural Atoms](https://arxiv.org/abs/2311.01276) combines
[GatedGCN BN](https://github.com/tmlr-group/NeuralAtom/blob/main/2D_Molecule/graphgps/layer/gatedgcn_layer.py)
with [LN-based neural-atom modules](https://github.com/tmlr-group/NeuralAtom/blob/main/2D_Molecule/graphgps/network/custom_gnn.py).

An additional recent counterexample, [DGT, Nature Communications 2026](https://www.nature.com/articles/s41467-026-75005-9),
explicitly applies BN in each graph transformer layer. It is not a verified
PCQM champion or a matched pure-2D control.

[Scholkemper et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6c473e69ba261200dd595d07494c1a73-Abstract-Conference.html)
separate rescaling, centering and initial-feature reinjection in linearized
GNN theory. Their GraphNormv2 is not simply renamed LayerNorm, and the theory
does not establish that K1 suffers oversmoothing or needs BN removal.

This targeted review does not establish a field-wide usage frequency. It does
refute the universal claim that recent molecular models have all abandoned BN.
The official PCQM leaderboard mixes geometry/pretraining/ensembles and hidden
test-dev scores; it is not a pure-2D, from-scratch, common-budget ranking.

## What existing MolGap evidence establishes

- [Frozen 500K calibration](../pcqm_k1_500k_bn_calibration/terminal_decision.md):
  unchanged learned parameters with clean BN buffers improve MAE by 1.245 meV.
  BN state matters; this does not establish that BN replacement improves training.
- [Dropout mechanism](../pcqm_k1_bn_mechanism_a100/terminal_decision.md): dropout
  during buffer estimation is worse than clean estimation. This is not evidence
  that dropout during weight training should be removed.
- [160-pass LN comparison](../pcqm_k1_norm_exposure_100k/terminal_decision.md):
  direct LN is 5.771 meV worse than same-job BN. Low-LR plus warmup nearly recovers
  the BN score across jobs, but does not establish superiority.
- [RMS mechanisms](../pcqm_k1_norm_mechanisms_100k/terminal_decision_20261023.md):
  pre-RMS loses to post-RMS; graph-scoped RMS is a positive exploratory signal,
  below the frozen 3 meV gate. Those variants replaced both local norm positions.

Consequently, neither 'BN is useless' nor 'LN/RMS cannot work' is justified.
Normalization type, position, statistics scope and residual/optimizer trajectory
have been conflated in informal explanations. More epochs alone did not rescue
the direct-LN recipe.

## New local discriminators

One original-BN reference and two isolated candidates use identical fixed 100K
training / 50K internal-development roles, initialization, seed and 40-pass recipe:

1. FFN-only node RMS: replace nine `norm3` modules, preserve all message-path BN.
   This distinguishes placement from the previously tested whole-block changes.
2. Channel-wise local residual gains initialized to one: preserve all BN and
   initialize the predictor identically; learn message/FFN branch scaling.
   This transfers a residual-control idea, not a published K1 recipe.

No combined candidate, LR grid, new geometry, protected role or remote submission
is authorized by this local screen. A negative result closes only its frozen
screen; a positive point estimate is not a seed-robust or 500K transfer result.
Protocol and observed execution status are owned by the adjacent files.
