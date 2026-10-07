# K1 consistency ablation at 500K

Desktop-owned question on `codex/exp/k1-consistency-ablation-500k`, based on
verified desktop `a386a4a2c1ba9ce3a841fb775dec0d88dada7458`.
Read [protocol](protocol.md), [evidence review](plan.md), and [status](STATUS.md).
Preparation reuses the retained bounded 500K trainer, source/release/planning
owners and Kaggle adapter. The registered 100K workflow does not execute this
500K family. Shared code provenance is in [reuse](reuse.md).

The two arms use the same retained pretrained K1 state and two dropout
forwards. Only the disagreement coefficient differs: 0 versus 0.1. No original
K1 reference is retrained. Each arm has an independent prospective trajectory;
terminal acceptance must produce independent trace, role, cost and V5/RML
records. Replay qualification is checked rather than assumed.

Run `prepare.py --declare-only` before committing source/declarations, then
`prepare.py --output <fresh-directory>` at the frozen executable HEAD.
Kaggle submission and continuation remain with the platform skill.
