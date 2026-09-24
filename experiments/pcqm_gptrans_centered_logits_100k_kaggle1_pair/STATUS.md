# Kaggle1 paired 100K status

Kernel: `nothingnessvoid/molgap-gptrans-centered-logits-paired-100k-s42-v1`.
Kaggle1 version 1 reported `KernelWorkerStatus.COMPLETE`. The desktop verified
the frozen source and graph identities and retained only the seven required
files per arm under
`platforms/_records/kaggle/training/pcqm_gptrans_centered_logits_100k_kaggle1_pair_v1/raw/`.

Both arms passed no-inference mechanical acceptance and completed separate V5,
cost, role, 60-epoch canonical-trace, and RML terminal transactions. The same-job
50K development comparison gave GPTrans-T `0.156014488 eV` and centered logits
`0.155416209 eV`; the `0.000598279 eV` gain misses the `0.003 eV` shortlist
floor, and the paired 95% interval crosses zero. The candidate is
`NEGATIVE_UNDER_CONTRACT`; the reference control is `CLOSED`. Official
validation and test roles remain sealed. `decision.md` and `results/` own the
terminal evidence.

RML frozen validation passes, but neither new trace is replay-pool complete.
Both are explicitly excluded because the frozen historical reference binding
cannot satisfy the current strict causal replay entrance for this control arm.
This is an evidence-qualification blocker, not a failed training run. Do not
change the frozen prospective references or relaunch this screen to fill it.
