# Precision comparison launch decision

The user requested a Kaggle1 training-speed and MAE comparison between FP32
and BF16 mixed precision. The accepted Kaggle1 runtime manifest identifies
Tesla T4 (compute capability 7.5); native CUDA BF16 requires capability 8.0.
After this limitation was explained, the user selected FP32 versus FP16 mixed
precision on Kaggle1. This is a new training-recipe question. A matched FP32
control is necessary to estimate the precision change under the same T4x2
allocation, data, seed, optimizer schedule and selection role.

The release is limited to one paired fixed-100K attempt after both local and
remote preflight pass. Neither a queue response nor a speed estimate is a
scientific result. The numerical decision is owned by `protocol.md` and the
terminal acceptance, not by this launch record.
