# Resource-Bounded Architecture Refresh Decision

Decision date: 2026-08-24

## Question

Can a from-scratch encoder architecture improve the repaired-2M PubChemQC
B3LYP model under a 12-hour limit for one training run?

This is an architecture-only review. It excludes pretrained checkpoints,
self-supervised pretraining, fine-tuning, distillation, warm starts, 3D teachers,
and multi-stage coordinate prediction. Both pure-2D and end-to-end 2D+3D
architectures were in scope. Frozen-encoder late 3D fusion was not. At the
review date, the frozen production comparator was `repaired_2m_dense_2d`.
This review did not change the registry or authorize a remote run. Live status
belongs in `CURRENT_STATE.md`.

## Observed architectural gap

The production GPS encoders use raw 9-dimensional atom features, 4-dimensional
bond-type features, GINE local message passing, dense global attention, and
mean pooling. Atom degree is already present. They do not use random-walk or
Laplacian structural encodings, shortest-path attention bias, persistent edge
states, a graph token, or learned attentive pooling.

This matters because the official GraphGPS recipe treats positional/structural
encoding, local message passing, and global attention as three separate parts.
Its PCQM4Mv2 models use RWSE. The official leaderboard also lists a
`GraphGPS without PE (subset)` result far below the full GPS entries, although
that row is not a controlled apples-to-apples ablation and must not be quoted as
the expected gain in MolGap.

The closed ring/conjugation adaptor is not contradictory evidence. That branch
tested whether routed-v4's worst residual decile was enriched in hand-selected
ring and conjugation descriptors. It did not train a from-scratch positional
encoding inside the encoder and did not test RWSE.

## Why the previous 2D+3D model regressed

The rejected repaired-2M fusion was not an end-to-end multimodal encoder. It
froze three 2D models and two SchNet models, concatenated predictions, weights,
prediction differences, and two 176-dimensional SchNet embeddings, normalized
that context on the internal training split, and learned a three-output
`0.10 * tanh(MLP(context))` correction on top of the 2D result.

That head improved the internal scaffold-disjoint test by `0.004102 eV` over
equal GPS7/GPS9 and `0.002455 eV` over dense three-GPS. On the fixed external
1,973-molecule comparison it instead regressed by `0.023251` and `0.024239 eV`.
The prediction audit identifies the failure mode:

| Residual system | Internal mean absolute correction | External mean absolute correction | External/internal | External p95 absolute correction |
|---|---:|---:|---:|---:|
| Equal GPS7/GPS9 + dual SchNet | 0.026800 | 0.070671 | 2.64x | 0.099995 |
| Dense three-GPS + dual SchNet | 0.020703 | 0.068947 | 3.33x | 0.099995 |

For the dense system, `58.2%` of LUMO corrections and `46.8%` of Gap
corrections had absolute magnitude at least `0.09 eV`; `88.6%` of LUMO
corrections were negative. The external context therefore drove the bounded
head into saturation rather than making small geometric refinements.

Five design choices explain the regression:

1. The last correction layer was not zero-initialized and had no confidence
   gate or correction penalty, so the exact 2D identity was available but not
   preferred.
2. Train-split feature normalization fed shifted external embeddings to an
   unconstrained MLP. The final `tanh` bounded the damage but did not prevent
   saturated extrapolation.
3. Standalone SchNet was materially weaker than the 2D ensemble. A large
   embedding can still fit residual correlations internally without providing
   a reliable external correction direction.
4. Both SchNet branches were evaluated on the same primary conformer. Their
   training histories differed, but inference did not supply two independent
   geometric observations.
5. ETKDGv3+MMFF geometry is approximate. On QM9, replacing ETKDG geometry with
   DFT geometry improved fusion by `0.02428 eV`, while six-conformer averaging
   recovered only `0.00511 eV`. Most of the geometry deficit is systematic, not
   random conformer noise.

This evidence rejects the previous late residual architecture, not 3D as a
modality. On QM9, GPS9 plus SchNet improved average MAE from `0.08419` to
`0.07653 eV`; the problem is how geometry entered the repaired-2M system.

## Top-20 feasibility screen

| Leaderboard family | Pure 2D from scratch? | Exact model under 12 h? | Transferable architecture idea |
|---|---|---|---|
| TGT / triangular EGT | No under this protocol | No: two large stages and staged coordinate training | Triplet interaction is rejected for this budget |
| Uni-Mol+, GraphGPT-3D, MolNet, Global-ViSNet, Transformer-M, GEM-2 | No | No | Put pairwise geometry inside shared transformer blocks rather than in a late molecule-level MLP |
| GraphGPT MLM | Pretraining-dependent | No | Excluded by protocol |
| GPS++ | Partly; official system also uses richer inputs | No: IPU-specific 44M model and 112-model final ensemble | Deeper hybrid blocks and richer structural input |
| GPTrans | Yes | Tiny is 6.6M, but official training uses 8 GPUs and 300 epochs | Persistent node-edge propagation, shortest-path/multi-hop encodings, graph token |
| Deep graph transformer | Yes | Exact 63.6M ensemble is too risky | Sparse deep residual attention and attentive pooling |
| EGT | Yes | No: 47-89M dense edge-channel models | Persistent edge channels, but quadratic memory is high |
| GraphGPS | Yes | Exact deep model takes about 60 h on one A100 | RWSE is the highest-confidence transferable change |

## Shortlist

### P0a: Structural GPS9-192

Keep the accepted GPS9 compute shape and training protocol, but add a learned
RWSE branch derived only from `edge_index`:

- random-walk return probabilities for steps 1 through 16;
- a small MLP that projects RWSE into the 192-dimensional node state and is
  summed with the atom embedding;
- the existing nine GINE plus global-attention blocks;
- first screen RWSE with existing mean pooling, then test attentive pooling only
  if RWSE is positive.

This was the highest-priority candidate. It reuses the immutable 2D graph cache, adds
little parameter cost, remains ordinary PyTorch/PyG, and directly fills a
missing part of the published GPS architecture. Based on the accepted GPS9
run (`3.736M`, `5.47 h`), the full repaired-2M estimate is `5.8-7.0 h` on the
same SCNet class, subject to a measured one-epoch projection.

### P0b: Geometry-biased Structural GPS9-192

Use P0a as the 2D backbone and insert geometry into every global-attention
block instead of adding a frozen SchNet tower:

- retain atom features, bonds, RWSE, GINE local message passing, and nine
  192-dimensional blocks;
- encode pairwise ETKDGv3+MMFF distances with a small radial basis expansion;
- combine radial features with bond flags and capped shortest-path distance to
  produce a per-head attention-logit bias;
- share one node state across 2D and 3D information and train the complete model
  from scratch with one three-target head;
- keep train and inference conformer construction identical.

The attention operation is already dense in the existing GPS block, so adding
a compact pair bias is much cheaper than executing two SchNet encoders. The
expected full-run cost is approximately `1.2-1.5x` GPS9, or `6.5-9.0 h` on a
single A100-class device after the graph cache exists. PyG's stock `GPSConv`
does not expose arbitrary pair bias, so this candidate needs a small reusable
attention block in `src/molgap/`; the first gate is a forward/memory/timing
check, not a full training submission.

This was the preferred 3D hypothesis because it copies the transferable common
idea behind Transformer-M and Uni-Mol+: persistent atom/pair interaction inside
the encoder. It deliberately omits their pretrained checkpoints, coordinate
refinement, triangular updates, and very large widths.

### P1: GPTrans-lite

Use one compact persistent node-edge transformer:

- eight layers, node width 192, edge width 24, four heads;
- shortest-path/multi-hop encoding capped at 12;
- node-to-edge, edge-to-node, and node-to-node propagation;
- graph token and LayerScale;
- one three-target B3LYP head.

This is the strongest genuinely different 2D hypothesis, but it maintains
dense pair states. MolGap molecules extend to MW 1000 and may be much larger
than typical PCQM4Mv2 molecules, so memory and wall time must be measured before
scale-up. It belongs on the single A100, not on Kaggle P100/T4 or SCNet DCU for
the first compatibility run. The full run is allowed only if a one-epoch timing
projection is at most 10 hours; otherwise stop at the screen.

### P2: Sparse DeepGraphTransformer-lite

Use 12 layers at width 192 with sparse `TransformerConv`, DeepGCN residual
wrappers, and attentive global pooling. It is portable and avoids dense pair
states, but the leaderboard's strong test result came from an ensemble while
the reported single validation MAE was weaker. Treat this as a diversity probe,
not the first scale-up candidate.

### P3: Gated radial Structural GPS

If P0b is memory-safe but not positive, add one lightweight radial message
branch per GPS block:

`h_out = GPS_block(h) + sigmoid(gate) * radial_message(h, distance)`

Initialize the radial gate near zero so the model starts as the 2D identity and
must earn geometric influence during joint training. This is a distinct
architecture test, not a repair of the rejected molecule-level residual head.
Its projected cost is `1.5-2.0x` GPS9 and it is permitted only if a measured
epoch remains below the 12-hour limit.

## Compute assignment

| Platform | Appropriate work |
|---|---|
| SCNet BW-1 DCU | Pure-2D Structural GPS and sparse DeepGraphTransformer after local import/forward checks |
| IMS single A100 | Geometry-biased GPS, GPTrans-lite dense-pair screen, and any full run that passes the timing gate |
| Kaggle P100/T4 | 100K scaffold-disjoint screens and independent seeds; no full dense-pair run |
| Colab A100 | Backup for a winning screen or full Structural/Geometry-biased GPS run with Drive checkpoints |

Every remote run must use atomic per-epoch checkpoints, resumable input shards,
an elapsed-time stop before the platform limit, and independently downloadable
metrics and predictions.

## Controlled experiment

All encoders are initialized randomly and trained directly on B3LYP labels.
No stage may load an existing model state.

1. Reuse the existing 100K scaffold-disjoint architecture-screen split and add
   a freshly trained GPS9-192 control. Split identity is owned by
   `../pubchemqc100k_architecture/results/experiment_manifest.json`; do not
   regenerate or reinterpret it.
2. Run three seeds for P0a (RWSE only). Test RWSE plus attentive pooling only if
   RWSE is positive. Run one feasibility seed each for P1 and P2.
3. On exactly the same 100K molecules and split, run P0b using the accepted
   ETKDGv3+MMFF protocol. Compare it against both fresh GPS9 and P0a; do not
   train a late fusion head.
4. Advance an architecture only if its mean validation average MAE improves the
   same-run GPS9 control by at least `0.001 eV`, all three P0 seeds agree in
   direction, and the one-epoch full-2M projection is at most 10 hours.
5. Train only the winner once on repaired-2M. Evaluate the standalone encoder on
   the frozen common, OOD, and P8-hard blocks before training any new ensemble.
6. Call it an architectural breakthrough only if it improves the production
   three-GPS dense comparator (`0.097638 / 0.106655 / 0.088407 eV`) by at least
   `0.001 eV` on common average MAE without regressing OOD or P8-hard by more
   than `0.0005 eV`. A weaker but complementary encoder may be retained as
   research evidence, but it does not change production.

## Sources

- OGB PCQM4Mv2 leaderboard: <https://ogb.stanford.edu/docs/lsc/leaderboards/#pcqm4mv2>
- GraphGPS official implementation: <https://github.com/rampasek/GraphGPS>
- GPTrans official implementation: <https://github.com/czczup/GPTrans>
- GPS++ official implementation: <https://github.com/graphcore/ogb-lsc-pcqm4mv2>
- Deep graph transformer submission: <https://github.com/daxiongshu/PCQM4Mv2_subs>
- EGT official implementation: <https://github.com/shamim-hussain/egt_pytorch>
- TGT official implementation: <https://github.com/shamim-hussain/tgt>
- Transformer-M official implementation: <https://github.com/lsj2408/Transformer-M>
- Uni-Mol+ official implementation: <https://github.com/deepmodeling/uni-mol>

## Local evidence

- External repaired-2M rejection: `../repaired_2m_scaling/results/hierarchical_dual_schnet_external/decision.md`
- Internal fusion metrics: `../repaired_2m_scaling/results/hierarchical_dual_schnet_v1_remote/metrics.json`
- Residual saturation audit: `fusion_failure_audit.json`
- Geometry ceiling and conformer averaging: `../qm9_architecture/results/conformer_scaling/decision.md`
- ETKDGv3+MMFF protocol decision: `../conformer_protocol/results/decision.md`
- PubChemQC 100K split contract: `../pubchemqc100k_architecture/results/experiment_manifest.json`
