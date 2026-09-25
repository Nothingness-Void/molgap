# Prelaunch infrastructure correction, 2026-09-25

The first source release (`paired_release.json`) passed the unmodified V4
reference preflight. Its reference process finished only epoch 0 and was
interrupted. Its checkpoint and trace are retained under the first release's
`reference/training` output; they are **not** a terminal model or part of the
paired comparison.

Inspection before the candidate launch found that the existing noisy-model
constructor supplied a frozen GPTrans-T core state to a strict whole-model
loader. The candidate adds two `denoise_head` parameters, so this would fail;
removing the frozen state instead would break the matched initialization
contract. The shared frozen-state loader now permits an explicitly named pair
of missing auxiliary keys while SHA-verifying the complete loaded core.
Real-file model construction and `11` focused tests passed.

`paired_release_v2.json` is the sole training release for the matched pair.
It has a new source commit/archive identity and fresh output paths. Both arms
must restart from epoch 0 on that release; neither the old partial reference
checkpoint nor a historical candidate checkpoint may be substituted. The
scientific data, seed, schedule, selection, roles and cost ceiling remain as
specified in `paired_training_contract.md`.
