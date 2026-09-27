# GPTrans local inductive bias 100K status

**Kaggle1 attempt 001 ended `KernelWorkerStatus.ERROR` before training.** The
remote `rwse16` worker accepted its T4 runtime preflight. The
`rwse16_local_edge` worker failed the frozen initial-model-state identity check:
observed `96eefae91355e36eff3566e063bee88bceae903ed8cae56e8269cc984b156cc9`,
expected `9e2e58fa54f68c061f47637a9c406dd4c849ae9b0d49623881f63bb94655db0b`.
The paired runner stopped before launching either training worker. See the
[attempt record](../../platforms/_records/kaggle/training/pcqm_gptrans_local_inductive_bias_100k_s42_v1/attempt_001/README.md)
and retained remote log.

Neither arm has a completion manifest, selected model, aligned development
predictions, resumable training checkpoint, or 60-epoch trace. There is no MAE
or B-versus-A scientific decision, and neither arm is replay-ready. The two
prospective RML trajectories remain open for the unresolved question; this
failed remote attempt does not count as negative mechanism evidence. The
[release decision](launch_decision.md) authorized only one T4x2 attempt and no
automatic retry. This experiment branch remains the owner pending a separately
reviewed next action.
