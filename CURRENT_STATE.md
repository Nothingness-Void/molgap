# Current State

> Live truth only. Historical methods, metrics, and failures live in experiment
> decisions; task order lives in `ROADMAP.md`.

## Production

- Recommended Track A model: repaired-2M three-GPS dense pure 2D.
- Registry: `repaired_2m_dense_2d`; lower-cost preset:
  `repaired_2m_equal_2d`.
- Loader: `load_repaired_2m_2d` in `src/molgap/inference.py`.
- Authority: `production/04_evaluate/project_freeze/track_a_final_decision.md`.
- Track B experiments cannot change this registry.

## Track B candidate

Neural-Atom K1 and the adapted GPTrans-T core are the two frozen architectures
in the desktop-owned matched full-run comparison. K1 replaces dense per-layer global GPS attention with
three exchanges through one 64-channel molecular slot while retaining local
persistent real-bond EdgeState processing. Its PCQM-100K, once-read shadow, and
fixed 500K bridge passed; evidence is in:

- `experiments/pcqm_k1_shadow/decision.md`
- `experiments/pcqm_k1_scale500k/decision.md`

K1 is frozen at 3,658,817 parameters and architecture commit
`36215d9539acdd75542608637ec1e2db5341d3ff`. It is not a production promotion,
an official leaderboard result, or evidence against the separately configured
desktop 304-wide/pretrained model.

GPTrans-T is frozen at 5,246,817 parameters. Its 100K reference was weak, but
the same propagation core passed the accepted SCNet 500K gate and was
numerically slightly better than K1's separate 500K run. Because their
architecture-specific optimization contracts differ, the full matched study—not
the cross-job 500K scalar difference—owns the final comparison. Authority:
`experiments/pcqm_gptrans_t_500k/results/decision.md`.

The audited full-role implementation is `experiments/pcqm_k1_full/`. The
desktop-owned branch `codex/pcqm-k1-gptrans-full-fusion` has launched a matched
K1/GPTrans comparison at 20M presentations, seed 42, strict FP32/no-TF32, and
physical BS128. Its recoverable checkpoints and scheduler recovery remain
desktop-owned; server must not launch a duplicate or interfere. Official-role
use and any final submission remain governed by that experiment's frozen
contract.

## Discovery state

The separately authorized three-round K1 architecture sequence is complete.
Round 1 closed RepSet final readout and cross-layer selector sharing. Round 2
found a directional but sub-threshold gain from removing length-one slot
attention. Round 3 found a similarly directional but sub-threshold gain from
uniform normalized return; inverse-score return regressed. No candidate cleared
the frozen `0.003 eV` material/run-variation gate, so K1-v4 remains unchanged.
Authority: `experiments/pcqm_k1_return_allocation_100k/decision.md`.

The explicitly reopened interaction screen combining the two directionally
positive K1 simplifications completed with a favorable but sub-threshold gain.
It is closed and does not alter the frozen K1 handoff. Authority:
`experiments/pcqm_k1_combined_simplification_100k/decision.md`.

The separate Kaggle2 discovery loop remains bounded to at most three
evidence-gated rounds. Round 1 completed a pure-2D GPTrans-T reference on the
accepted 100K/50K data identity. It was mechanically accepted but substantially
underperformed the contextual K1 reference. Reconciliation with the accepted
500K run does not isolate a causal scale effect: exposure, EMA time scale and
evaluation roles differ. The user explicitly reopened the two unused Kaggle2
rounds. The first completed: Pair PreNorm passed the mechanism-shortlist gate;
Centered Logits regressed and is closed. No K1/full promotion follows.
Authority: `experiments/pcqm_gptrans_relation_flow/decision.md`.
The last authorized round completed: both persistent-pair readback variants
failed to improve the reference and are closed. That two-round authorization
was exhausted without a full-scale promotion.
Pair PreNorm is the only retained seed42 mechanism shortlist, not a full-scale
or K1 promotion. Authority and consolidated evidence:
`experiments/pcqm_gptrans_memory_readback/decision.md`.

The user reopened three additional evidence-gated Kaggle2 rounds. Round 1
closed isolated K1 real-bond storage/read normalization: one arm regressed and
the other was directionally favorable but below the material gate. Round 2
tested that weak relation-recurrence signal with the two favorable global-slot
simplifications. Neither pair passed the material gate or retained both parent
effects, so triple stacking is closed. One authorized round remains, pending a
fresh evidence audit rather than another micro-variant. Authorities:
`experiments/pcqm_k1_edge_memory_100k/decision.md` and
`experiments/pcqm_k1_edge_slot_interaction_100k/decision.md`.

The bounded three-attempt K1 mechanism sequence is complete. Paper-style
multi-slot grouping, channel-wise multi-head atom selection, and a
molecule-conditioned single query all fit training more tightly and generalized
worse under otherwise matched v4 contracts. Their decisions are
`experiments/pcqm_k1_paper_allocation_100k/decision.md`,
`experiments/pcqm_k1_multiview_pool_100k/decision.md`, and
`experiments/pcqm_k1_dynamic_query_100k/decision.md`. No attempt qualifies for
another seed, shadow access, scale-up, official evaluation, or submission.

The bounded K1-v4 selective-global screen completed with no winner. K1-G was
materially worse; K1-R was statistically indistinguishable and slightly worse.
Both are closed without extra seeds or scale-up. The accepted K1-v4 reference
remains reusable only for its exact benchmark contract. Authority:
`experiments/pcqm_k1_variants_100k/decision.md`.

GraphState and its derivatives failed full-scale transfer and are archive-only.
The local-hierarchy allocation, GAPE, adaptive denoising, cardinality channel,
multi-slot Neural-Atom, K1-G, K1-R, Fourier Edge, geometry, path, ring, PNA,
directed-bond, fragment, dual-stream, and attention variants are closed by
their own records.
The consolidated attribution is
`experiments/pcqm_gap_architecture/results/architecture_failure_attribution_2026-09-08/decision.md`;
archive indexes are `experiments/_closed/pcqm_server_archive_index.md` and
`experiments/_closed/qm9_top20_archive_index.md`.

SCNet experiments and full training are desktop-owned and excluded from the
server queue. The new screen does not modify the frozen K1 full handoff.

## Data and comparison contract

- PCQM4Mv2 fixed 100K, 500K, 1M, and full identities are accepted under
  `platforms/_records/ims/pcqm_fixed_datasets_v1/`; Kaggle mirrors are indexed
  under `platforms/_records/kaggle/`.
- New screens use one immutable baseline per scientific contract. A candidate
  may compare directly across platforms without retraining that baseline when
  data, row order, seed, FP32 mode, BS128, optimizer, schedule, loss, selection,
  exposure, and role access match and each runtime has one reusable calibration
  certificate. Authority: `experiments/SCREENING_POLICY.md`.
- Historical paired-v3 results retain their original contracts. The K1 500K
  result remains valid but its 32-row epoch tail prevents use as a new v4
  reference.
- A reused development score is selection evidence, not an unbiased final
  estimate. Official validation and test-dev remain sealed.

## Hard boundaries

- Track B predicts Gap directly on official PCQM4Mv2; Track A remains on the
  repaired-2M PubChemQC corpus. No silent dataset replacement or augmentation.
- No teacher/distillation or privileged geometry on the OGB leaderboard line.
- Any geometry used in training and inference must use the same ETKDG contract.
- Molecular-research-server access is separately gated by
  `platforms/REMOTE_HANDOFF.md` and limited to `/lustre/home/users/sm2/chou/`.
- Remote jobs require immutable cache acceptance, atomic checkpoints,
  retrievable outputs, and a dated decision. Scientific failures close a route;
  only infrastructure failures may retry unchanged.

## Pointers

| Question | Authority |
|---|---|
| What happens next? | `ROADMAP.md` |
| What ships? | `production/README.md` |
| Track meanings | `TRACKS.md` |
| Experiment index | `experiments/README.md` |
| Remote operations | `platforms/README.md` and `platforms/REMOTE_HANDOFF.md` |
| Code ownership | `ARCHITECTURE.md` |
| Artifact inventory | `models/README.md` |
