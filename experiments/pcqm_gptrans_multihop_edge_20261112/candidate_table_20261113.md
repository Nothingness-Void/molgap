# Candidate structure comparison table (2026-11-13)

Planning record only. No training, no Kaggle quota, no protected-role read.
It does not amend `protocol.md` and authorizes nothing. Every candidate still
needs a prospective `trajectory.json` before any new diagnostic or training
(AGENTS.md, "RML and Research Entry").

Evidence was collected by read-only Haiku delegates on 2026-11-13 and is
second-hand. `W` = this worktree, `M` = `D:/文档/molgap`, `E` = this experiment
directory, `HE` = `M/experiments/pcqm_k1_complete_local_audit/historical_evidence.md`,
`RV` = `M/platforms/_records/local/research_reviews/edgestate_20261006/report.md`.

## 1. Ruling

No candidate has enough evidence for immediate training. Rank by expected
decision value per T4 hour:

| Rank | Candidate | Ruling | One-line reason |
|---|---|---|---|
| 1 | D. Training-only DFT-distance auxiliary head | Run free diagnostics; needs user approval for a new data source | Only candidate that adds information absent from every tested run |
| 2 | G. Joint 2D/3D training with mode dropout (Transformer-M style) | Hold until D is positive | Largest literature effect, same data dependency as D, more code |
| 3 | C. Triangle (edge-to-edge) update in the GPT-T pair stream | Run free diagnostics | Names a real missing path in GPT-T; repo triplet variants stayed under 3 meV |
| 4 | F. Third decorrelated member judged by fusion gain | Free simulation only | Heterogeneous fusion is the largest measured effect in the repo |
| 5 | A. Nested GNN (rooted ego subgraphs) | Free collision count, then likely NO_TRAIN | Gap unproven: K1 has RWSE16, GPT-T has full-pair SPD |
| 6 | E. SPSE / ring encoding | NO_TRAIN by arithmetic | Literature effect 0.7-0.8 meV needs about 21 pairs to detect |
| 7 | B. Small TokenGT as a single model | NO_TRAIN | Drops SPD bias; 48.5M TokenGT is already worse than 6M GPTrans-T |

## 2. What the two current models see

GPT-T (E1/E2 recipe: BondLocal + pair_update_norm + multi-hop path term):

- Nodes: OGB 9 atom features (degree, aromatic, IsInRing) plus in/out degree
  embedding (`W/src/molgap/ogb_features.py:12,45-57`, `W/src/molgap/gptrans.py:161-162,248-253`). No RWSE (`gptrans.py:286`).
- Pairs: bond embedding on bonded pairs only; SPD embedding capped at 20, with
  "beyond 20" and "unreachable" sharing one bucket (`gptrans.py:143,211,223,260-267`).
  The multi-hop term sums bond embeddings along one chosen shortest path
  (`W/src/molgap/gptrans_multihop_edge.py:11,20,27-40`).
- Flow: full-pair attention; node-to-pair, pair-to-node and pair-to-bias exist
  (`gptrans.py:63-85`). There is no edge-to-edge or triangle update
  (`gptrans.py:6-7`).
- Readout: graph token plus virtual pair (`gptrans.py:292-293`). 12 layers,
  node 256, pair 32, 8 heads, about 5.89M parameters, batch 128, fp32.

K1 (`neural_atom_k1`):

- Nodes: OGB 9 features plus RWSE16 (`W/src/molgap/gps.py:134-138`,
  `W/src/molgap/qm9_neural_atom.py:217-218,254`). No degree embedding.
- Edges: bond state updated on real bonds only. No SPD, no non-bonded pair state.
- Flow: 9 local `ResGatedGraphConv` blocks, no local attention; global mixing
  only through single-slot `NeuralAtomMixer` at layers 3, 6, 9
  (`qm9_neural_atom.py:86-93,98-196,231-244`).
- Readout: mean pool only (`qm9_neural_atom.py:224`). 3,658,817 parameters
  (`W/experiments/pcqm_k1_full/protocol.md:12`).

Information that is lost or merged (delegate inference from code, not measured):

| Gap | GPT-T | K1 |
|---|---|---|
| g1. Number of shortest paths between a pair | invisible; one path is read | implicit in RWSE only |
| g2. Bond types on alternative paths (conjugated vs not) | only the chosen path | local only, no pair-level code |
| g3. Pair-to-pair consistency (i,k),(k,j) -> (i,j) | absent | absent |
| g4. Long-range pair identity | present up to SPD 20 | weak: about 9 hops plus 3 global slots |
| g5. Equilibrium geometry (target is defined on DFT geometry) | absent | absent |
| g6. 1-WL-equivalent ring layouts | separated through SPD | separated through RWSE |

No repo evidence shows that any of g1-g6 explains current error. The existing
residual breakdown only says the highest ring-fraction quartile favours K1 by
4.698 meV (`M/experiments/pcqm_500k_v4_evidence/residual_attribution.md:14-38`).

## 3. Noise, pair counts and cost

- N1, 100K, four ref inits, dev (fixed100k-next50k, best-development-EMA):
  138.513, 139.287, 140.009, 142.218 meV. Range D = 3.705 meV, sample SD
  S_ref = 1.596 meV (`E/decision_e1_n1.md:28`).
- E1 paired deltas: +0.658 and -0.667 meV, mean -0.004 (`E/decision_e1_n1.md:12-18`).
  These are two deltas, not a measured spread.
- P1 (seeds s50-s54) is running; S_ref, S_pair and m7 are pending
  (`E/protocol.md:159-168`). The numbers below must be recomputed when P1 lands.
- Pair-count rule in force: `2*sqrt(2)*S_ref/sqrt(n) <= effect` (`E/protocol.md:165`).
  Arithmetic by the adjudicator with S_ref = 1.596 (n = 4, so itself uncertain):

| Effect to detect at 100K | Pairs, pairing cancels nothing | Pairs, if P1 shows S_pair <= 0.5*sqrt(2)*S_ref |
|---|---|---|
| 3.7 meV | 2 | 2 (floor) |
| 3.0 meV | 3 | 2 (floor) |
| 2.0 meV | 6 | 2 |
| 1.0 meV | 21 | 6 |

- A new model family (A, B) cannot share a frozen init with GPT-T or K1, so the
  left column applies to it whatever P1 shows. Addons on GPT-T (C, D, G) can
  use same-init pairs.
- Measured cost of one GPT-T 100K arm on T4: 2.37-2.59 process hours
  (`E/closure_*/measured_*.json`). One pair is about 5 T4 h. GPT-T peak GPU
  memory: not found. Per-pass seconds: not measured.
- Graph size: mean about 13.9 atoms, largest per-batch maximum 18 in the 100K
  set (`W/experiments/pcqm_input_pipeline/analysis.json:5-12`). Per-molecule
  maximum: not found.
- Transfer: 100K gains shrank to 24-37% at 500K (PairNorm -4.84 -> -1.14,
  Noisy Nodes -4.84 -> -1.81) or changed sign (PairToken +3.04 gain -> -0.70;
  receiver/triplet/RRWP gains 1.9/1.5/2.4 -> regress 2.2/1.9/3.6)
  (`M/experiments/pcqm_gptrans_pair_norm_500k/decision.md:23`,
  `M/experiments/pcqm_gptrans_noisy_nodes_500k/decision.md:12-24`,
  `M/experiments/pcqm_k1_complete_local_audit/analysis_zh.md:187-192`).
  Planning consequence, not a gate change: a candidate whose plausible 100K
  effect is under about 8 meV is unlikely to keep 3 meV at 500K.

## 4. Master table

Expected gains are the adjudicator's priors. Literature numbers came from the
user's research summary; no such summary file exists in the repo, so they are
unverified here.

| | A. Nested GNN | B. Small TokenGT | C. Triangle update (TGT style) | D. DFT-distance aux head (train only) | E. SPSE / ring encoding | F. Third fusion member | G. Joint 2D/3D, mode dropout |
|---|---|---|---|---|---|---|---|
| Question | Does a rooted k-hop subgraph per atom separate atoms that K1 merges? | Do bond tokens attending to each other beat pair bias? | Does letting pair (i,j) read (i,k),(k,j) recover g1-g3? | Does supervising pair states with DFT distances supply g5? | Does an explicit path or ring count supply g1? | Can a cheap, different model lower fused error even if weaker alone? | Does 3D input on part of the training batches improve 2D-only inference? |
| Gap named | g6 for K1 | g3 on bonds only | g1, g2, g3 in GPT-T | g5 | g1 | none; targets residual correlation | g5 |
| Repo evidence of the gap | none | none | none measured; code confirms the path is absent | none measured | none | fusion: K1+GPT dev residual corr 0.8395, blend -4.261 meV vs K1; tail hetero -4.38 meV vs best single | none measured |
| Nearest tested | K1 hop shell 100K, -1.116 meV, NEGATIVE; MoSE31 rooted counts replacing RWSE, NEGATIVE; MoSE residual gate -2.046, below gate | K1 PairToken 100K +3.04 gain, sign flip at 500K; QM9 graph token R7, closed; TokenGT rejected on parameter cap | GPT-T triplet-return norm 100K -0.62 / +0.65, NEGATIVE; K1 sparse triplet +1.12 worse; K1 triplet PairToken -2.61 and triplet-aggregate -1.52 below gate, regress at 500K; GPT-T pair memory NEGATIVE; TGT rejected on budget | GPT-T Noisy Nodes (atom-type reconstruction) 100K -4.84, 500K -1.81 NEGATIVE; feature denoising +2.1 to +3.4 worse; ETKDG distance-angle input 500K -3.52 | E1 multi-hop edge, NEGATIVE; RRWP 100K gain 2.45 then regress 3.58 at 500K; SPD pair-token bias -1.67 below gate; MoSE as in A | K1+GPT fusion and distillation lines; same-family seeds corr 0.875-0.906 | none; distance-angle used ETKDG as an inference input, a different contract |
| Difference from tested | Independent subgraph encoders; tested items changed features or shell order inside one encoder | Full token model without SPD; tested items added one pair token to K1 | Dense per-layer triangle update in the GPT-T pair stream; tested items were sparse, one-shot, K1-side or a normalization | Uses official DFT coordinates as a label; every tested auxiliary used 2D-derived or ETKDG signals | Counts all simple paths; tested items read one path or random-walk terms | Selects by fused gain, not single MAE | 3D enters as input in training only; inference stays 2D |
| Expected 100K gain | 0-2 meV | negative as a single model | 2-5 meV | 2-6 meV | under 1 meV | unknown until simulated | 4-10 meV if literature transfers |
| Basis | no PCQM evidence; RV:96-102 rates GNN-AK+ low priority | TokenGT 91.0 meV at 48.5M vs GPTrans-T 83.3 (unverified) | TGT vs EGT literature gap (unverified, larger models, uses a distance stage); repo triplet items all under 3 meV | Transformer-M ablation: 3D denoising about 2.4 meV, joint pretraining about 9.7 (unverified, full data) | SPSE paper 0.7-0.8 meV (RV:88-95) | measured fusion gains above | Transformer-M joint pretraining about 9.7 meV (unverified) |
| T4 / 16 GB bound | About 14 subgraphs per molecule, each near molecule size: roughly 10x node count per batch; estimated, not measured | About 30 tokens per molecule; cheap | Triplet term is batch x N^3 x heads; at N = 18, batch 128, 8 heads about 6M entries per layer: fits; time 1.5-3x estimated | One extra head and an N x N target; negligible | SPSE precompute about 80 h (RV:88-95) | one extra 100K run per member | Extra distance-bias encoder; about 1.1-1.3x estimated |
| 100K pairs needed | new family: 6 or more for 2 meV | none; no training proposed | 3 for 3 meV, 6 for 2 meV; same-init pairing possible | 3 for 3 meV, 6 for 2 meV; same-init pairing possible | about 21 for 1 meV, over 100 T4 h | n/a | 2-3 if the effect is 4 meV or more |
| Minimal build | one addon: k = 2 ego batches through shared K1 blocks, pool to root | not proposed | one GPT-T addon: axial triangle attention on the 32-d pair state in the top 4 layers | sidecar file of DFT pair distances keyed by row; binned-distance head on the pair state; loss off at inference | not proposed | none | D plus a Gaussian-basis distance bias with random drop |
| Main risk | cost with no named gap; molecules are so small that k = 2 covers most of the graph | loses SPD | repeats the triplet pattern: small 100K gain that fades at 500K | acts as a regularizer that fades at 500K like Noisy Nodes; data and contract work | effect below noise floor | fused gain does not make a better single model | train/inference mismatch; most code of all candidates |
| Free diagnostic (no training) | Count 100K training atoms that share a 1-WL + RWSE16 signature but differ in rooted 2-hop subgraph hash, and the label spread inside those classes | parameter and exposure count only | On saved dev predictions, compare error for pairs-rich structure: fused rings and molecules with multiple shortest paths vs the rest | (1) confirm the official training SDF is obtainable and row-aligned; (2) on saved dev predictions, test whether error rises with rotatable-bond count | none needed | From saved predictions, compute fused MAE as a function of a hypothetical member's MAE and residual correlation | same as D |
| Stop rule for the diagnostic | under 0.1% of atoms affected: NO_TRAIN | n/a | no error excess in those strata: NO_TRAIN | SDF unavailable or no trend with flexibility: lower priority | n/a | required correlation below anything a small model reaches: close | follows D |

## 5. Next free steps, in order

1. D-diagnostic 2 and C-diagnostic share one pass over the saved 100K dev
   predictions (`D:/molgap-artifacts/retrieved/pcqm-gptrans-multihop-edge-20261112-{s43,s44,n1-s45}/v1/version1/*/development_predictions.pt`)
   with RDKit descriptors from SMILES: rotatable bonds, fused-ring count,
   shortest-path multiplicity. The 50K internal dev rows are training-set rows,
   not a protected role.
2. D-diagnostic 1: check whether the official PCQM4Mv2 training SDF is on disk
   or must be downloaded, and whether row order matches `full_train_v1`.
3. F simulation from existing tail prediction files; descriptive only.
4. A collision count on 100K training topologies.
5. Recompute section 3 when P1 lands.

## 6. Needs user approval

- Any training run for any candidate.
- D and G: downloading official DFT coordinates and a new contract for
  training-only 3D labels. `W/src/molgap/pcqm_graph_inputs.py:34-37` forbids
  `pos` in pure-2D inputs and AGENTS.md says "Pure-2D must not construct 3D".
  A sidecar must exclude the `tail_holdout_v1` rows and must not rebuild the
  fixed datasets.
- Opening a new `codex/exp/` branch and prospective trajectory for whichever
  candidate proceeds; the diagnostics in section 5 belong there, not here.

## 7. Unverified or missing

- All literature numbers (no summary file in the repo).
- Section 2 gap table: inference from code, not tested.
- GPT-T peak GPU memory, per-pass seconds, per-molecule maximum atom count.
- What "triplet-return" refers to in `M/experiments/pcqm_gptrans_triplet_norm_100k_20261104/decision.md:12-16`,
  given that `gptrans.py:6-7` excludes triplets.
- HE:202 records a full-data K1+GPT fusion as NEGATIVE while 500K fusion gains
  about 4.3 meV; the contracts differ and were not reconciled here.
- Whether `store/source/pcqm4m-v2.zip` contains coordinates (not opened).
- Court ruling on Nested GNN and TokenGT: supplied by the user; no record in the repo.

Side observations, not acted on:

- `gptrans.py:266-267` adds `virtual_pair` twice at position [0,0].
- `E/protocol.md:92` estimates a 1.415x slowdown for the multi-hop arm; measured
  process hours give about 1.08-1.09x.
- The 100K manifest format (`M/data/pcqm_fixed_100k_v1/manifest.json:2`) differs
  from the one `pcqm_graph_inputs.py:25,89-92,108` expects; the Kaggle reader
  path was not traced.
