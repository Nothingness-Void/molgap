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

Neural-Atom K1 is the sole frozen server-side candidate eligible for a desktop
full-run budget decision. It replaces dense per-layer global GPS attention with
three exchanges through one 64-channel molecular slot while retaining local
persistent real-bond EdgeState processing. Its PCQM-100K, once-read shadow, and
fixed 500K bridge passed; evidence is in:

- `experiments/pcqm_k1_shadow/decision.md`
- `experiments/pcqm_k1_scale500k/decision.md`

K1 is frozen at 3,658,817 parameters and architecture commit
`36215d9539acdd75542608637ec1e2db5341d3ff`. It is not a production promotion,
an official leaderboard result, or evidence against the separately configured
desktop 304-wide/pretrained model.

The audited full-role implementation is `experiments/pcqm_k1_full/`. It fixes
sample exposure at 20M presentations (156,250 BS128 steps), uses strict
FP32/no-TF32, atomically resumes all training state, requires an accepted
optimizer-inclusive runtime preflight, and emits a self-contained model bundle.
It is unlaunched and desktop-owned. No server full run, K1 tuning, extra seed,
official-validation read, test-dev read, or submission is authorized.

## Discovery state

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
server queue. This bounded screen does not modify the frozen K1 full handoff.

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
