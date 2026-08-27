# Resource-Bounded Pure-2D Route Audit

Audit date: 2026-08-28

## Scope

This audit ranks architecture mechanisms for the official PCQM4Mv2 direct-Gap
task under the MolGap constraints: official OGB categorical 2D graphs,
train-derived 100K selection, at most 5.2M parameters during screening, one GPU
task at a time, and a projected single-A100 full run below 12 hours. Official
validation and test-dev remain sealed.

## Evidence from completed MolGap routes

| Mechanism family | Local evidence | Disposition for the next PCQM question |
|---|---|---|
| Persistent real-bond state | Selected on QM9 and on matched PCQM Gap100K | Retain as the backbone |
| Dense all-pairs state and triplet repair | Slower and less accurate than sparse EdgeState | Closed |
| Final multi-depth or learned-query readout | Regressed on QM9 or PCQM | Closed |
| Static local-operator replacement | GatedGCN, TransformerConv, and GENConv all lost to persistent EdgeState | Closed |
| Edge-to-node FiLM | Close but negative on all QM9 targets | Closed |
| Recurrent graph state | Improved QM9 Gap but lost the old three-target average through LUMO regression | Highest-priority transfer to the now Gap-only PCQM task |
| Virtual multihop edges and shortest-path attention | Added no net information beyond RWSE plus GPS attention | Closed |
| Non-backtracking directed bond memory | Improved QM9 Gap slightly but lost the old three-target average | Second priority |
| PNA neighborhood statistics | Not yet run; literature-backed but with more compute and no project-specific positive result | Third priority |
| Gated retention inside the edge update | Not yet run; low-cost and mechanistically plausible but no positive empirical result | Fourth priority |

Exact metrics and gates remain in the linked decisions under
`experiments/top20_architecture_qm9/` and
`experiments/pcqm_gap_architecture/results/`.

## External architecture evidence

- The official PCQM4Mv2 task is direct HOMO-LUMO Gap regression evaluated by
  MAE. The official leaderboard shows that competitive graph Transformers are
  generally much larger than this project's budget:
  <https://ogb.stanford.edu/docs/lsc/pcqm4mv2/> and
  <https://ogb.stanford.edu/docs/lsc/leaderboards/>.
- GraphGPS supports the retained hybrid design: positional/structural
  encoding, real-edge local message passing, and global attention are separate
  ingredients: <https://proceedings.neurips.cc/paper/2022/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html>.
- GPS++, the PCQM4Mv2 challenge winner, explicitly maintains node, edge, and
  graph-level states in every block. Its report attributes most parameters and
  much of the efficiency to the expressive MPNN path, but its 44.3M single
  model and 3D/auxiliary components are outside this project:
  <https://arxiv.org/abs/2212.02229>.
- Directed bond memory is chemically credible and avoids immediate reverse-
  edge feedback, as used by D-MPNN/Chemprop:
  <https://pubs.acs.org/doi/10.1021/acs.jcim.3c01250>.
- PNA combines mean, maximum, minimum, and standard-deviation aggregation with
  degree scalers and reports strong molecular results, but it expands every
  local block: <https://proceedings.neurips.cc/paper/2020/hash/99cad265a1768cc2dd013f0e740300ae-Abstract.html>.
- EGT, TokenGT, and GRIT evolve dense pair states or relative-position channels
  and are strong research references, but their all-pairs cost and typical
  parameter scales conflict with the current single-card bound:
  <https://arxiv.org/abs/2108.03348>,
  <https://arxiv.org/abs/2207.02505>, and
  <https://proceedings.mlr.press/v202/ma23c.html>.

## Selection

The next candidate is `ogb_recurrent_graph_state_gps9`: the accepted OGB
EdgeState GPS9 plus one shared recurrent molecule state with a 16-channel
update bottleneck. The state is updated from mean-pooled nodes and broadcast
before every GPS block. Its output projections are zero-initialized, so the
initial function preserves the EdgeState path while optimization can learn a
global correction.

This route ranks first because it has all three forms of support unavailable to
the alternatives: positive Gap-specific project evidence, direct precedent in
a PCQM-specialized architecture, and negligible parameter/throughput risk. It
is a new PCQM Gap-only question, not an attempt to overturn the earlier QM9
three-target decision.
