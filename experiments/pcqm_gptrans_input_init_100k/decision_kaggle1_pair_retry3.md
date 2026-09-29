# Kaggle1 GPTrans-T input-initialization pair — terminal decision, 2026-09-30

The shared-runner Kaggle1 T4x2 attempt completed both 60-epoch arms. Local
mechanical acceptance verified the frozen 100K/50K manifest, 46,860 optimizer
steps and 5,998,080 sample presentations per arm, selected model and prediction
hashes, finite source-aligned development rows, recomputed MAE, runtime
certificates, candidate initialization hash and protected-role exclusions.
`results_kaggle1_pair_retry3/` owns the mechanical and paired result records;
`platforms/_records/kaggle/training/pcqm_gptrans_input_init_pair_retry3_v1/`
owns the retrieved remote evidence.

The same-job control scored **0.156014488 eV** and the Normal(0,0.02)
input-initialization candidate scored **0.157283103 eV**. The candidate was
**0.001268614 eV worse** on the 50,000 aligned development rows. A 10,000-draw
paired row bootstrap (seed 42) gave a 95% candidate-minus-reference interval
of **[0.000089655, 0.002449788] eV**. The frozen shortlist gate required at
least 0.003 eV improvement and an interval upper bound below zero; it failed.
The scientific outcome for this exact 100K intervention is
`NEGATIVE_UNDER_CONTRACT`, without a second seed, 500K, full-scale or protected
evaluation release. This does not establish that the published GPTrans module
is ineffective or that the adapted implementation has a bug.

Both arms used 5,246,817 parameters. The candidate's alternating same-device
preflight training-step and inference-forward median ratios were 0.909 and
0.995, below the 1.05 ceiling. The sums of the 60 observed single-T4 epoch
training/development intervals were 2.8464 and 2.8474 hours respectively;
these exclude setup and queue time. The row bootstrap does not estimate
training-seed variation. See `attribution_kaggle1_pair_retry3.md` for the
failure-mode interpretation and remaining limits.
