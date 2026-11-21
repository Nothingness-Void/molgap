# GPT-T with a train-only DFT distance head: protocol

Frozen 2026-11-21 JST before any diagnostic or training. Desktop-owned, Track B,
branch `codex/exp/gptrans-dft-distance-aux-20261121`.

## Contract class: 3D-supervised training, pure-2D inference

This is a **new contract that needs its own qualification**. Training reads the
DFT coordinates that ship with the official PCQM4Mv2 training set and uses them
only as an auxiliary label (all-pair interatomic distances). Model input and
inference stay pure 2D: no coordinates at inference, no 3D construction, no
ETKDG. Results are booked separately from strict pure-2D results and are never
ranked against them as if they were the same contract
(`research_memory/reviews/edgestate_single_model/report.md` lines 86 and 110).
User authority: conversation of 2026-11-21 (train-time DFT coordinates allowed
as auxiliary labels only). Not authorized and not done: inference-time label
lookup or correction.

## Evidence review (RML, 2026-11-21)

- No DFT-coordinate or 3D-teacher trajectory exists in desktop or
  `origin/molgap-server` RML (`lookup --mechanism dft|sdf|3d|distance|denois|geometry`).
- Nearest prior work, all different mechanisms:
  ETKDG geometry as **input** (`TB-gptrans-geometry-distance-*-a2` CLOSED /
  NEGATIVE_UNDER_CONTRACT; 60-epoch continuation +0.857 meV GPT-T), Noisy Nodes
  auxiliary (`TB-gptrans-noisy-nodes-100k-s42` POSITIVE, `-500k-s42` NEGATIVE),
  chemical descriptor/fingerprint auxiliary (INCONCLUSIVE), feature denoising
  (INCONCLUSIVE).
- **Main prior risk:** auxiliary objectives here have helped at 100K and then
  lost at 500K (Noisy Nodes). A 100K pass alone releases nothing; the 500K pair
  is the decision point.
- Literature basis (3D-supervised, coordinate-free inference): Transformer-M and
  related PCQM4Mv2 leaderboard entries; GPS++/TGT-style distance targets. These
  set no expected gain for GPT-T at this size.

## Hypothesis and falsifiers

- Observed deficiency: GPT-T must infer conformation-dependent electronic
  effects from topology alone; no training signal tells it about real geometry.
- Changed mechanism: a train-only head reads GPT-T's final pair state and
  regresses DFT heavy-atom distances for all atom pairs of a molecule. The Gap
  path, data order, dropout and initialization are unchanged.
- Alternative explanations for any gain: extra gradient signal acting as a
  generic regularizer (as Noisy Nodes did at 100K); clip-norm interaction;
  seed noise (about 1-2 meV at 100K).
- Cheapest falsifier (CPU only, before any GPU): the SDF must match the fixed
  graphs. For every row, heavy-atom elements and bonds of the SDF molecule must
  map one-to-one onto the fixed-store graph; bonded distances after mapping must
  lie in 0.9-2.0 A. NO_TRAIN if alignment fails for more than 1% of rows, if
  bonded distances fall outside that range for more than 0.1% of rows, or if
  building the 500K sidecar takes longer than about 7 minutes (2D build scale).
- Training falsifier: see gates below.

## Data and roles

- Source: `pcqm4m-v2-train.sdf.tar.gz` (official PCQM4Mv2 training geometries,
  9,746,078,962 bytes uncompressed member). The official host
  `ogb-data.stanford.edu` timed out from both the desktop and IMS on
  2026-11-21, so the file came from the Hugging Face mirror
  `liuganghuggingface/moldata-pcqm4mv2`, exact path
  `structures/pcqm4m-v2-train.sdf.tar.gz`, sha256
  `8690db7b573405e3dbd617482fe10092e18be2b86e4d8500c3892d301dd5817c`
  (equal to the mirror's LFS oid). No other mirror file was fetched. The mirror
  is third-party; its identity is proven by the row-by-row topology match above,
  not trusted. The official `data.csv.gz` is not read (it carries official
  validation labels); the reference is the accepted fixed-store graph.
- Coordinates are built only for official train rows 0..499,999 (the 500K train
  role, which contains the 100K train role). Development rows get no
  coordinates; tail_holdout_v1 rows are never parsed.
- Roles: 100K train 0..99,999, development 100,000..149,999; 500K train
  0..499,999, development 500,000..549,999. Development is the selection and
  gate role. Official valid/test and tail_holdout_v1 are not read.

## Arms (one IMS A100 job, serial chain)

| Item | Model | Scale | Init / seed | Budget (s) | Gate |
|---|---|---|---|---:|---|
| c100 | GPT-T BondLocal pair-update-norm (J2 recipe bytes) | 100K | s43 init, seed 43 | 4,140 | none |
| d100 | same + DFT distance head | 100K | s43 init, seed 43 | 4,800 | none |
| d500 | same + DFT distance head | 500K | s44 init, seed 42 | 17,600 | dev MAE d100 - c100 <= -3.0 meV |
| c500 | GPT-T control (J5 recipe bytes) | 500K | s44 init, seed 42 | 15,300 | same gate |

Total budget 41,840 s. Recipes are identical between control and candidate
except the mode and the sidecar. Head constants are frozen in the addon module:
loss weight 0.1 on the normalized Gap L1 scale, L1 loss in angstrom over all
real heavy-atom pairs of rows with valid coordinates, head initialized from its
own generator so the global RNG stream matches the control. One arm, no lambda
grid.

## Decision rules (after retrieval)

- 100K: d100 - c100 dev EMA MAE (selected) <= -3.0 meV and paired 2000-row
  bootstrap 95% upper bound < 0 -> 100K positive under this contract. Otherwise
  NEGATIVE_UNDER_CONTRACT at 100K; 500K items are skipped by the job gate.
- 500K: d500 - c500 <= -3.0 meV and bootstrap upper < 0 -> nominate both final
  checkpoints for one tail_holdout_v1 score each, booked in the separate
  3D-supervised ledger. Otherwise NEGATIVE_UNDER_CONTRACT (gain did not survive
  scale) or INCONCLUSIVE (between 0 and -3.0, or interval crosses 0).
- Tail and dev scales are never compared with each other. No adoption,
  full-scale run or production change follows from this protocol.
- Inference check before any tail score: the selected candidate checkpoint must
  score with no sidecar present and no coordinate field in the batch.
</content>
</invoke>

## Deviations (dated, before any training)

- 2026-11-21: the bonded-distance sanity proxy (0.9-2.0 A) was replaced by an
  element-aware rule and a bond-order length check; see
  `diagnostic_alignment_20261121.md`. The pre-submit review found the v1
  sidecar's atom mapping ignored bond order (swapped =O/-OH and similar in
  about 7.6% of rows); v1 is withdrawn and a chemistry-labelled v2 sidecar
  replaces it. No arm, constant, role or training gate changed.
- 2026-11-21: budgets set from the local rehearsal (candidate slower per
  step); c500 runs before d500 so a time pause falls on the candidate, which
  resumes. Budgets are estimates used for time skips, not limits on training.
- The in-job 500K gate uses the point estimate only. The protocol's 100K
  positive also needs the bootstrap upper bound < 0, checked after retrieval;
  a 500K run that the job launched does not by itself make 100K positive.
- c100 and c500 are byte-identical to j2 and j5 of job 2097505 (same recipe,
  init and seed). Their extra A100 time is booked to this question; if both
  jobs run, the pair also serves as a determinism check.
