# GPTrans precision comparison status

The desktop-owned FP32 versus FP16 comparison was submitted to Kaggle1 as one
T4x2 kernel. Kaggle assigned the canonical run ID
`nothingnessvoid/molgap-gptrans-fp32-fp16-precision-100k-s42-v1`; the first
authoritative status query returned `RUNNING`. The requested slug ending in
`gptrans-fp16-precision-100k-s42-v1` was normalized by Kaggle. The launch
receipt and observed ID are under [launch/](launch/README.md).

Both arms have prospective RML trajectories. Remote preflight, training,
measured T4 speed, development MAE, terminal acceptance and replay readiness
remain pending. No official validation or test role is authorized.
