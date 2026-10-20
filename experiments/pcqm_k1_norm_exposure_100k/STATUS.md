# Status

Authorized on 2026-10-21 JST: two Kaggle1 T4x2 notebooks, four arms.
Owner: `codex/exp/k1-norm-exposure-100k`, forked from desktop `66fbb13d`.

## Submitted jobs

Observed at 2026-10-21 02:12 JST through the authenticated Kaggle1 API:

| Pair | Physical kernel/version | Arms | Scheduler state | Receipts |
|---|---|---|---|---|
| norm | [139859525 / v1](https://www.kaggle.com/code/nothingnessvoid/molgap-k1-ln-norm-160ep-100k-s42) | `bn_long`, `ln_long` | QUEUED | [norm release](norm_release1/submission/) |
| optimization | [139859934 / v1](https://www.kaggle.com/code/nothingnessvoid/molgap-k1-ln-optim-160ep-100k-s42) | `ln_low_lr`, `ln_low_lr_warmup` | RUNNING | [optimization release](optimization_release1/submission/) |

Both physical version1 source entries, private visibility, explicit T4 request,
mounted source/data identities and uploaded source bytes were verified by the
existing journaled submitter. RUNNING is a scheduler observation, not native
qualification or proof of optimizer progress. Startup log sampling returned no
events; native preflight and formal training remain unobserved locally.

## Frozen execution

Source commit: `69c91dbcd7df22804f977067e78982bad43a89a0`.
The subsequent host-only Kaggle discovery fix does not change either payload.

| Pair | Spec identity | Package identity |
|---|---|---|
| norm | `965ff813b52af17cf88ecb8653a73adab402e2ec72f1d2a8f7807609128a81d7` | `4cec8bf05cecc9e44aca73bfe309cd3846e814fe4114a004debbfc4bd2163f51` |
| optimization | `8230927a983f1a79b6a34190ef2360e136cd960ea8003dc7530e6fdcf66859f9` | `39b2d7edb51310e04dd57204a37ba80e959f936934c876da14c95727089b9194` |

The [protocol](protocol.md) owns recipes, 160-pass exposure, curve endpoints,
resource limits, roles and acceptance. Four prospective RML trajectories were
published before submission and the derived state passed the frozen check.
Terminal mechanical/scientific acceptance and strict replay qualification are
pending independently for every arm.

Prepared packages and full local operation records are retained under
`D:/molgap-artifacts/runs/<logical_id>/v1/`; each release directory owns its
logical ID and the small copied submission receipts. No training output was
downloaded during submission. No automatic continuation or monitor was created.
