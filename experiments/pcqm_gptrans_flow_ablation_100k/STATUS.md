# Operational status

Kaggle1 kernel `nothingnessvoid/molgap-gptrans-flow-ablation-s42` version 1
completed both 60-epoch arms. Frozen no-inference acceptance passed all source,
data, runtime, checkpoint, prediction, alignment, and hash checks.

Both deletions regressed against immutable GPTrans-T: `no_pair_to_node` reached
`0.1592841231 eV` and `no_pair_recurrence` reached `0.1581743161 eV`, versus
`0.1566272043 eV`. Both paired intervals were entirely unfavorable. The
trajectory is closed as `NEGATIVE_UNDER_CONTRACT`; no scale-up or protected
role was released. Authority: `decision.md` and `results/acceptance_v1.json`.
