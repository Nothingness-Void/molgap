# Start checks evidence (retrieved 2026-11-17)

- Preflight and `[arm]` lines of both jobs: `log_excerpt_s42.txt`, `log_excerpt_s43.txt` (excerpts of `logs/train.log` in each root; every task shows `preflight ok | torch 2.7.1+cu126 | NVIDIA A100-SXM4-80GB | rows 500000` and the `[arm]` line, `batch=1024 lr=2.828e-03 steps/epoch=488 total_steps=29,280`).
- The scheduler `.o` files (`GPTW500S42.o2025095`, `GPTW500S43.o2025099`) are 0 bytes; all output went to `logs/train.log`.
- `login_check` output: the raw output of the pre-submit login-node run was not saved on the remote roots or locally, so it cannot be archived. Its result is only the summary in `ims_submission_width_500k.md` ("strict load ok" for all four arms of both cards, expected parameter counts). Re-running it would be compute on the login node and was not done.
