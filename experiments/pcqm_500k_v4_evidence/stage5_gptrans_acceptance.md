# Stage 5 GPTrans-T terminal acceptance

Kaggle kernel `nothingnessvoid/molgap-500k-v4-gptrans-s42` version 7
completed on 2026-09-16. The frozen no-inference `accept_stage.py` check was
rerun locally against the retrieved raw evidence and accepted the result.

- Arm: `gptrans`.
- Contract fingerprint: `ddbfdd8d9f88a7315efad1fd446f96da61052488a823848c1302baed6d3efd4a`.
- Completed exposure: 60 epochs, 234360 optimizer steps, 29998080 sample presentations.
- Best internal-development Gap MAE: `0.10686753690242767 eV` at zero-based epoch 56.
- Final epoch internal-development Gap MAE: `0.107020802795887 eV`.
- Final epoch training MAE: `0.09482762640340961 eV`.
- Stage-manifest SHA256: `7cf5243d0a5a34818f8a86801347be9dffb6fd6b81a66df34958c9582a79c536`.
- Last-checkpoint SHA256: `1a74aaad2e3e08da9cad8bdef642e4ca2b2b77c1d240b658049877994a70ae06`.
- Best-model SHA256: `e311f1972ba74ade32e1f16be717c4484825eeafbafd592758775553445c79e`.
- Best-predictions SHA256: `0608084eb2a7a90923a652235572ee69578c4e138d5df3c457e5170f7da613bf`.
- Runtime-certificate SHA256: `e69bf5fc401b609898cec441793c9a6e4221d84dbf0ecded934364980e3e1e8e`.
- Runtime: one Tesla T4, PyTorch `2.4.1+cu121`, deterministic FP32, TF32 disabled, seed 42.
- Official validation, test-dev and test-challenge role-read flags remained false.

This record accepts the GPTrans-T arm mechanically. It does not decide the
matched three-arm comparison. The contemporaneous EdgeState/K1 version 6
kernel was still RUNNING at inspection, so paired residual/bootstrap analysis
and the nomination decision remain blocked on those two accepted terminal
artifacts. No successor, extra seed or official-role evaluation was authorized.

Raw evidence:
`platforms/_records/kaggle/training/pcqm_500k_v4_stage5/gptrans-v7/`.
