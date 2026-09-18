# V5 Route B Trajectory Audit

Audit date: 2026-09-19

This is a documentation audit, not a new scientific run. It maps the local
PCQM/Route B evidence to the V5 trajectory-lite shape:

`state -> evidence -> action -> result -> decision`

The V5 evidence envelopes currently present in the repository are pointer-only
wrappers around older records. They are useful for integrity and role
accounting, but they are not yet a complete trajectory ledger.

## Route Map

| Local route | State | Evidence | Action | Result | Decision | V5 status |
|---|---|---|---|---|---|---|
| `pcqm_gap_architecture` recurrent graph state | Frozen 100K Gap-only screen; one seed authorized | Acceptance, decision, summary, trace, and launch manifest under `results/recurrent_graph_state_seed42/` | One Kaggle seed-42 run | Completed and missed the frozen EdgeState comparator by `0.000630125 eV` | Closed without seed 43/44 or scale-up; archived at commit `285e1dc` | **Complete negative / closed** |
| `pcqm_edge_state_full` | Historical full official-only EdgeState baseline | `v5_evidence.json` plus rich-full acceptance and IMS manifest | Full training, official validation, and one external submission preparation | Official-valid Gap MAE `0.102063 eV`; external result remains a specialist reference | Closed; no resubmission; no production transfer | **Complete envelope** |
| `pcqm_gptrans_t_100k_v4` | Frozen 100K/50K V4 reference | `v5_evidence.json`, training contract, acceptance | One fixed reference run | Development Gap MAE `0.156627 eV` | Closed for 100K promotion; reusable as context only | **Complete envelope** |
| `pcqm_500k_v4_evidence` | Matched 500K V4 three-arm comparison | Final decision, paired comparison, runtime and acceptance records | K1, GPTrans-T, and EdgeState under one contract | K1 `0.104860`, GPTrans-T `0.106868`, EdgeState `0.111349 eV` | K1/GPTrans-T nominated over EdgeState; K1-versus-GPTrans-T not material; no promotion | **Envelope missing** |
| `pcqm_geometry_transfer_500k` | Historical geometry nomination with an incompatible V4 bridge | `v5_evidence.json`, fusion metrics, comparability audit | Distance/angle additions to GPTrans-T and K1 | OOF blend `0.099546 eV`, but runtime/reference equivalence was not qualified | Hold; transfer blocked; do not scale up from this record | **Complete but blocked** |
| `pcqm_k1_gptrans_full_fusion` | Full official-train pair and fusion | `v5_evidence.json`, model/fusion acceptance, durable IMS artifacts | Full K1, full GPTrans-T, and fixed fusion study | Blend `0.102186 eV`; it did not beat the `0.099638 eV` EdgeState reference | Closed without promotion | **Complete envelope** |
| `pcqm_gine_expert` | One-million-row Gap specialist | `v5_evidence.json`, local manifest and predictions | Local full specialist training and fixed official-valid read | Official-valid Gap MAE `0.184618 eV` | Accepted specialist only; no leaderboard or production promotion | **Complete envelope** |
| `pcqm_route_b` legacy GPS/SchNet route | Frozen 1M four-encoder specialist record | `results/run_plan.json`, graph acceptance, fusion decision | GPS9, GPS11-160, two SchNet branches, bounded residual fusion | Fixed-valid Gap MAE `0.112011 eV` | Track B specialist; no registry change | **Envelope missing** |
| `pcqm_gptrans_t_500k` historical bridge | Pre-V4 500K context | `results/acceptance.json` and decision | Adapted GPTrans-T versus historical comparator | GPTrans-T `0.103948 eV` | Context only; not a V4 reference | **Legacy-only** |

The important distinction is that a route may be scientifically closed while
its V5 migration is incomplete. Missing an envelope does not invalidate the
underlying result; it only prevents the result from being consumed by a V5
agent without reading several legacy files.

## Missing V5 Modules

### 1. Trajectory-lite ledger

There is no shared machine-readable object that connects the five V5 stages.
`v5_desktop.py` validates evidence envelopes and desktop decisions, but it does
not store the hypothesis, action, result classification, or terminal reason.
`experiment_db.py` stores models, evaluations, artifacts, and causes; it does
not represent an experiment trajectory.

The missing object should be small and pointer-only. It should contain:

- `trajectory_id`, track, question, model/checkpoint identity, and contract ID;
- a hypothesis card: baseline deficiency, direct evidence, alternative
  explanation, one mechanism, cheapest falsifier, related closed routes,
  estimated native cost, and the decision it could change;
- the five stages `state`, `evidence`, `action`, `result`, and `decision`;
- separate execution, artifact, comparison, scientific, transfer, budget, and
  handoff outcomes;
- role-use history and pointers to immutable metrics, predictions, manifests,
  and runtime certificates;
- a terminal classification such as `NO_TRAIN`, `NEGATIVE_UNDER_CONTRACT`,
  `INCONCLUSIVE`, `INFRA_FAILURE`, `STOP_FOR_COST`, or
  `DUPLICATE_EVIDENCE`.

This should be a reusable validator/ledger utility, not a scheduler. The V5
desktop contract intentionally forbids desktop heartbeat and server takeover.

### 2. Native-cost ledger

Runtime numbers exist in individual records, but there is no normalized ledger
for load/collate, H2D, forward, backward/optimizer, validation, checkpoint,
archive, queue, preflight, failed allocation, retry, and acceptance time. The
device unit is also not consistently separated from wall time.

The ledger must preserve native units separately for A100, T4/P100, DCU, CPU,
queue hours, and wall hours. An unknown value must remain explicit rather than
being inferred from another platform.

### 3. `READY_FOR_DESKTOP` package

The policy function in `v5_desktop.py` can decide that a candidate is eligible,
but the repository has no generator and validator for the durable package
described by V5. A package should point to the frozen source/configuration,
accepted 100K/500K evidence, aligned predictions, role history, recovery
schedule, runtime certificate, native cost, and the exact desktop decision
required. It must stop server scale-up; it must not submit anything itself.

### 4. Cross-route role and reuse index

Each envelope has local `role_use`, but there is no one compact index answering
which official validation, test-dev, challenge, or development role has been
consumed by which route. This is the main reason a new agent still has to open
several decision files before it can rule out a duplicate run.

### 5. Calibrated trajectory trace

The repository has learning traces, but not a same-contract comparison of early
and late prefixes that can identify slow starters or justify an early stop. A
future cost-saving rule must be backtested on completed traces and must resume
the same optimizer, scheduler, RNG, and data cursor; it cannot restart a new
schedule and call that a trajectory continuation.

## Literature Extension

The local literature review already covers GraphGPS, GPS++, EGT, PNA, TGT, and
related graph-transformer families. The following additions make the evidence
usable for Route B without treating leaderboard models as drop-in code.

| Paper / mechanism | What it contributes | Local compatibility | V5 disposition |
|---|---|---|---|
| [GPS++](https://arxiv.org/abs/2212.02229) | Hybrid local MPNN/global Transformer, node-edge-graph state, 3D positions, auxiliary denoising; the reported system uses a large IPU-oriented model and ensemble | The graph-state idea matches the recurrent-state question; the exact 44M/112-model recipe does not | Reuse as a mechanism hypothesis, not a reproduction target |
| [ViSNet](https://www.nature.com/articles/s41467-023-43720-2) | Vector-scalar interactive geometric message passing with rotationally meaningful 3D interactions; evaluated on PCQM4Mv2 | `src/molgap/visnet.py` exists, but the local A/B record was much slower than SchNet and is not a Route B V5 contract | Keep as a geometry-limit diagnostic only; require a fresh 100K Gap-only card and native-cost gate |
| [EGT](https://arxiv.org/abs/2108.03348) | Layer-persistent pair/edge channels that evolve with global attention | The project already has EdgeState and GPTrans-style pair paths; dense EGT remains memory-heavy | Treat as prior evidence for persistent pair state, not a reason to reopen closed dense routes |
| [Graphormer](https://arxiv.org/abs/2106.05234) | Centrality, shortest-path, and edge-aware attention bias | RWSE and shortest-path features already cover part of this idea | Only test a clean missing bias mechanism; do not combine several old mechanisms in one screen |
| [DimeNet](https://arxiv.org/abs/2003.03123) | Directional messages, angular information, spherical basis functions | No PCQM Route B implementation or matched contract; the mechanism is more expensive than radial SchNet | Research reference; no full Route B run under the present card budget |
| [GemNet](https://arxiv.org/abs/2106.08903) | Directed edge embeddings and two-hop angular/dihedral message passing | The directed-edge idea relates to the closed QM9 EdgeState work, but a full GemNet path is outside the 12-hour bound until profiled | Keep as a geometry ceiling reference; only a bounded preflight could be justified |
| [GeoMFormer](https://proceedings.mlr.press/v235/chen24ac.html) | Transformer-based geometric molecular representation with explicit geometric interactions | Useful for a later geometry-attention design, but no local PCQM contract or cost certificate exists | Literature-only until a single mechanism and falsifier are registered |

The most useful conclusion from the literature is not “copy the best model.” It
is that the local routes already cover the major 2D information-flow families:
RWSE/GPS, persistent real-bond EdgeState, graph-state updates, and compact
pair propagation. The uncovered scientific question is narrower: whether a
geometry encoder can improve direct Gap under the same data, split, conformer,
and native-cost contract. The local closed A/B record makes ViSNet a weak
first choice; DimeNet/GemNet are higher-information but higher-cost candidates.

## V5-Compatible Action Order

1. Preserve the closed recurrent graph-state result as negative evidence; do
   not reopen seeds 43/44 or scale-up without a materially new information-flow
   question and an explicit contract.
2. Migrate the legacy `pcqm_route_b`, `pcqm_500k_v4_evidence`, and historical
   GPTrans bridge records into pointer-only V5 trajectory envelopes. This is
   documentation work and consumes no GPU hours.
3. Add the trajectory-lite validator, native-cost ledger, and role/reuse index
   as shared utilities. Keep scheduler control on the server branch and keep
   desktop code policy-only.
4. Do not launch a new 3D architecture from the papers until an existing
   geometry deficit is demonstrated under a matched V5 contract. If that gate
   opens, compare one mechanism only: a compact ViSNet/TensorNet-style native
   geometry encoder or an in-block radial/pair bias, not a late residual head.

No item in this audit changes the production registry, reopens sealed roles, or
authorizes a new remote job.
