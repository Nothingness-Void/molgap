# GPTrans precision comparison status

Kaggle1 reports the canonical kernel
`nothingnessvoid/molgap-gptrans-fp32-fp16-precision-100k-s42-v1` as
`KernelWorkerStatus.COMPLETE`. The returned source commit, package identity, and
graph manifest match the frozen launch pins. Minimal per-arm completion,
preflight, runtime, model, prediction, checkpoint, and trace artifacts plus a
small diagnostic log are retained under `remote_output/`.

The paired runs completed 60 epochs, 46,860 steps, and 5,998,080 presentations
per arm. FP16 took 9.43% longer in synchronized optimizer-loop time and did not
meet the frozen 20% speedup gate. The paired development MAE difference was
-0.00024337 eV (95% row-bootstrap interval [-0.00120294, 0.00072650]); the
interval crosses zero. Scientific decision: `NEGATIVE_UNDER_CONTRACT`; no FP16
benefit is nominated. See [decision.md](decision.md) and
[results/terminal_observation.json](results/terminal_observation.json).

Formal V5 acceptance and dual replay readiness remain blocked by the missing
precision-specific acceptance adapter and the frozen prospective same-run
replay binding. The exact gaps are recorded in the terminal observation. The
prospective trajectories remain unfinalized; no terminal RML entries or replay
pool entries are claimed. Protected evaluation roles remain sealed.
