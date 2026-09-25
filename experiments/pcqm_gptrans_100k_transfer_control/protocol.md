# Local GPTrans 100K transfer control: preflight

Question: can the local RTX 5060 run an accepted-source GPTrans baseline and
Noisy Nodes + Pair Update Norm control pair cheaply enough to separate
100K development-cohort effects from a 500K training-horizon effect?

This initial action is a throughput preflight, not a model comparison.
Use only accepted official-train-derived graph rows `0:100000` from the
SHA-pinned 500K V4 cache and perform a short FP32/no-TF32 batch-128 training
step benchmark. Do not read any development, official-validation, test-dev, or
challenge label or prediction. Benchmark elapsed wall time and GPU memory;
state separately what is estimated for the full 46,860-step/60-epoch V4
reference, and what overhead remains unmeasured. Do not interpret the
benchmark weights as a checkpoint or a model result.

The 500K cache source is the retained accepted
`D:/文档/molgap-exp/molgap-500k-v4-evidence/data/cache/pcqm4mv2_500k_v4`
manifest, canonical SHA256
`630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`.
The benchmark must verify the first two train-shard SHA256 values against that
manifest. Local hardware/time measurements are not Kaggle T4 device hours.

No full training is released by this preflight. Before running a scientific
pair, freeze a separate prospective action with exact source/init/config,
train and development row identities, target transform, raw/EMA rule, matched
schedule/steps, checkpoint/recovery plan, role-use record, native cost ceiling,
and stop rule. A separately trained historical reference cannot silently
replace the paired reference. Test-dev/challenge remain sealed.
