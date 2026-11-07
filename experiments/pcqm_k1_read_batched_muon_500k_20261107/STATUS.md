# Preparation status

2026-11-07 JST: documentation and unchanged current-read implementation only.
No input/recipe freeze, prospective publication, diagnostic, training or remote
submission has occurred for this question.

Owner: `codex/exp/k1-read-batched-muon-500k-20261107`, based on verified clean
desktop `eca56da4ca5e7a26040714252219fc11318a919f`.
Account: Kaggle1 `nothingnessvoid`; kernel/job/source version remain unassigned.

Before source freeze, integrate the reviewed independent registration of
`k1-muon-batched-adamw-fp32-v1`; generate its execution policy with the actual
builder. Generate one self-contained control recipe and metadata-only data
declaration with its consumed recipe hash. Author the real input with complete
family3 model declarations for both arms, no shared training override of the
reference, and an explicit500K target manifest. Publish prospective trajectories
before any new diagnostic/training. Native batched-execution qualification is
pending and cannot reuse old unbatched Muon qualification as its proof.

The original AdamW/Muon500K owner and its Kaggle2 job143011302/version2 are
independent and must not be modified or used as a strict comparator here.
