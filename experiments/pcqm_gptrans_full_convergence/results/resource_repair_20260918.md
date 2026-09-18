# IMS convergence resource repair, 2026-09-18

The K1 and GPTrans convergence continuations did not fail from CUDA memory or
model instability. Both jobs consumed exactly 15,296 MB of host memory and
ended with `Cgroup mem limit exceeded`. Neither job reached the first new
26,395-step official-validation evaluation, so the unchanged best metrics do
not provide scientific evidence against further continuation.

The accepted atomic checkpoints were preserved and verified before reuse:

- K1: continuation step 18,075, SHA256
  `a8a4c02868d419b21e0ae9a5d280c303c8fb5a60829c30249d976b63db1900db`.
- GPTrans: continuation step 19,812, SHA256
  `a686c6e1d12666a4dca6dac1b0472175863228a0fb0ae7b663bb465fbd90abb6`.

The successor jobs change only the execution envelope: 16 CPU cores, explicit
GPU job type, one GPU, a 72-hour scheduler limit, and a 70-hour application
budget. Physical batch 128, FP32/no-TF32, optimizer state, RNG state, data
order, model state, learning-rate schedule, validation interval, and stopping
rule are unchanged. Acceptance now exits cleanly when a durable paused
checkpoint exists without a completion manifest.

Submitted chains:

- K1: `1543823.ccpbs1` -> `1543824.ccpbs1`.
- GPTrans: `1543825.ccpbs1` -> `1543827.ccpbs1`.

Exact machine-readable provenance is in `../repair_submission_r5.json`.
