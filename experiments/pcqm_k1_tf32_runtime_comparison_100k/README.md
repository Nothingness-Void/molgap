# K1 FP32 versus TF32 on IMS A100

Question: does A100 TF32 matrix multiplication materially reduce the runtime
of the otherwise unchanged fixed-100K K1 screen, and what happens to its
internal-development Gap MAE? The answer is an execution diagnostic, not a
new architecture or a replacement for the desktop full-run contract.

The frozen scientific and operational conditions are in `protocol.md`.
`run.py` is the thin scheduler entry point; `accept.py` independently checks
retrieved outputs without model inference. Live state belongs in
`CURRENT_STATE.md` after a job is actually submitted.
