# Current State

This file owns the project recommendation, blockers and queue observations, not
evidence. [ROADMAP](ROADMAP.md) owns priorities; [BRANCHES](BRANCHES.md) owns
checkout routing; the [lifecycle map](docs/operations/LIFECYCLE_MAP.md) owns the
command order; [skills/](skills/README.md) own remote operations.

## Production Identity
- Track A recommendation: repaired-2M three-GPS dense pure 2D (registry `repaired_2m_dense_2d`; lower-cost preset `repaired_2m_equal_2d`).
- Public loader: `load_repaired_2m_2d` in `src/molgap/inference.py`.
- Authority: [Track A decision](production/04_evaluate/project_freeze/track_a_final_decision.md); [asset hashes](models/README.md).

## Track B
Track B predicts PCQM4Mv2 Gap only; it does not change Track A. Accepted full
EdgeState is the operational full-scale reference; the
[scale reassessment](experiments/pcqm_scale_transfer_reassessment/decision.md) owns the
limits of comparing 500K and full contracts. The [GPTrans 100K reference](experiments/pcqm_gptrans_t_100k_v4/decision.md)
is reusable but closed for 100K promotion. Official validation and test roles have
recorded consumption; reuse needs explicit authority ([reference roles](models/REFERENCE_INDEX.md)).
**For closed and open items by model family, read [families.md](research_memory/derived/families.md)**
(generated, never stale). A closed negative releases no successor, adoption, retry or scale-up.
Registered K1 and GPTrans runs use accelerated execution with no slow fallback
([K1](docs/operations/K1_EXECUTION.md), [GPTrans](docs/operations/GPTRANS_EXECUTION.md)).

## Blockers
- Missing frozen references stay pending, never retrained. Unknown remote state is `UNKNOWN`: reconcile first.
- TPU request metadata did not establish TPU execution; qualification stays pending under `platforms/_records/kaggle/preflight/`.
- No observation below releases adoption, scale-up, heartbeat or server takeover.

## Open Acceptance Items (owner branch holds the evidence)

| Item | State | Missing | trajectory_id | Branch |
|---|---|---|---|---|
| Real-tri-pair 500K (two Kaggle segments) | Both arms 60/60, retrieved; dev EMA delta cannot reach the frozen gate | Strict acceptance, role audit, closure | `T-pcqm-gptrans-real-tri-pair-500k-20261118-s42-real_tri_pair` | `codex/exp/gptrans-real-tri-pair-500k-20261118` |
| GPT-T width/FFN 500K (IMS) | Closed by decision; no variant nominated | Terminal RML record (placeholder `source_commit`) | `TB-pcqm-gptrans-width-500k-20261117` | `codex/exp/gptrans-width-500k-20261117` |
| Cutlass-GT 500K r2 (IMS) | Complete, retrieved, tail scored once; fails the tail bar | Decision and closure (the 100K screen is closed on its branch, not imported here) | - | `codex/exp/cutlass-gt-100k-20261118` |
| GRIT-SPSE 500K r2 (IMS) | Complete, retrieved, tail scored once; fails the tail bar | Decision and closure (the 100K screen is closed here) | - | `codex/exp/grit-spse-100k-20261118` |
| DeMol-CIN s43 IMS replication | User cancelled the four jobs; one serial job replaces them, not submitted | None | `T-pcqm-demol-cin-ims-replication-20261120-s43-demol_cin` | `codex/exp/demol-cin-ims-replication-20261120` |
| DeMol-CIN late give-back (local) | Arm A finished; arms B and C not run | Decision and closure for A; B/C need the `training_workers=0` prepare step | `T-pcqm-demol-cin-100k-20261118-r2-s42-demol_cin` | `codex/exp/demol-cin-late-giveback-20261119` |
| GPT-T DFT-distance input P1 (IMS, run `gptdftp1-chain-20261123`) | Continuation job 2161254 on the owner branch; gate (iii) stop rule is 0167ebb4a there | Retrieve, then score: check out the owner branch (`git archive` or temporary worktree) and run `run.py --repo-root <checkout> --script experiments/pcqm_gptrans_dft_input_p1_20261123/score_p1.py -- --retrieved <dir, flat or training/> --fixed-root ... --dev-sidecar ... --items ...`; the script is not on this branch | - | `codex/exp/gptrans-dft-input-p1-20261123` |
| Local single-forward/EMA 100K | Clean-EMA misses the frozen primary tolerance; cost crosses RTX 5060 and T4 | Terminal evidence, strict reference | `TB-pcqm-k1-single-true-ema-100k` | `molgap-desktop` |
| GPT-T triplet norm 100K | Six arms still ACTIVE in the index | Terminal closure | `T-pcqm-gptrans-triplet-norm-100k-20261104-s42-joint_norm` (+5) | `molgap-desktop` |

## Needs Check (last record is old; confirm on the owner branch first)

| Item | Last known state | trajectory_id | Branch |
|---|---|---|---|
| K1 mean2 vs single-forward EMA 500K | Kaggle1 stopped at the frozen cost cap after 29/53 passes; partial exposure is not a reference | `TB-k1-mean2-single-ema-500k-20261013-mean2` | `codex/exp/k1-mean2-single-ema-500k-20261013` |
| K1 execution pair custody | Fused and initialized layouts ran 40 epochs; speed gates failed; initialized layout adopted for new declarations | `TB-molgap-k1-fused-layout-100k-s42-v3-fused_layout` | `codex/exp/k1-fused-layout-t4-100k` |
| K1 bounded single/EMA 500K (T4) | Worker stopped at the cost cap; matched partial passes mechanical, science INCONCLUSIVE | `TB-k1-single-ema-500k-t4-20261010` | `codex/exp/k1-colab-500k-efficient` |
| Native T4 cost / 100K quality | 40/40 complete; native qualification passes, single fails quality; closure blocked by allocation timing | `TB-k1-t4-cost-quality-single-100k-s42-v1` | `codex/exp/k1-t4-cost-quality-100k` |
| Standalone K1 EMA 100K | Version 2 ERROR with empty outputs; cause unresolved | `TB-k1-weight-ema-ema999-kaggle3-100k-s42-v1` | `codex/exp/k1-ema-100k-night-20261006` |
| Gaussian spectral 100K | 40/40 complete; fresh-pair gate fails; closure pending | `TB-k1-spectral-night-spectral-kaggle3-100k-s42-v1` | `codex/exp/k1-spectral-100k-night-20261006` |

## Operating Boundaries
- V5 stays stable: [common contract](docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md), [desktop handoff](docs/operations/DESKTOP_AGENT_HANDOFF_V5_FINAL.md). No heartbeat, takeover or cross-machine live handoff.
- Remote work needs frozen identities, resumable checkpoints and durable outputs; IMS obeys [REMOTE_HANDOFF](platforms/REMOTE_HANDOFF.md).
- Common, OOD, P8-hard and official roles are not debugging or tuning data. Geometry keeps ETKDGv3+MMFF train/inference consistency.
- Router, MoE, replacement data and closed fusion need a genuinely new question.
- Old status text is in tag `docs-archive-20261124`.
