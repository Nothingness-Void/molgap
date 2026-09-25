# Accepted matched-500K trace snapshots

These four JSON files are exact copies of already accepted Kaggle V4 outputs
from the retained local `molgap-500k-v4-evidence` worktree. They contain only
stage metadata and epoch aggregates; no checkpoint, prediction tensor, label,
or protected evaluation data is copied here.

| Arm | Original retained record | Stage manifest SHA256 | Trace SHA256 |
| --- | --- | --- | --- |
| K1 | `platforms/_records/kaggle/training/pcqm_500k_v4_stage5/edge-k1-v6/evidence/neural_atom_k1/` | `6de0d6a576729030138a5f0a6914f7afe9cac75860b4355fe49c1e3c5e09f7e1` | `61fc6dda2395d2f93e591af6fe1dc213dd7f6ebadd541bc4c0a31240cf831630` |
| EdgeState | `platforms/_records/kaggle/training/pcqm_500k_v4_stage6/full-gps-final-epoch-v8/evidence/full_gps/` | `01f3862ff557eb3d46cd8901f7512f8bc33f9a10b8e000ce4c69b098fac05c51` | `f0908c6b9a984eb8cec473ee08836bbdd8191415374a9381a2636473a5561902` |

The K1 stage SHA is stated in the accepted
`experiments/pcqm_500k_v4_evidence/stage5_edge_k1_acceptance.md` record.
The EdgeState stage SHA is stated in the accepted
`experiments/pcqm_500k_v4_evidence/final_decision.md` record. Both manifests
declare their trace SHA, and `analyze.py` rechecks the bytes, complete
60-epoch status, contract fields, and exact optimizer-step/sample axes against
the canonical GPTrans matched-500K trace. This snapshot is analysis evidence,
not a new RML replay-ready reference or a new scientific training run.
