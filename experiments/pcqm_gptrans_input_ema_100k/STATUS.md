# Operational status

The submitted physical job is identified by [receipt](gpu/submission_v1.json).
Submission history and verification are in [submission record](gpu/submission_record.md).
Live repository truth belongs to `CURRENT_STATE.md`, not this experiment record.

For terminal saved-output acceptance and per-arm RML closure:

```powershell
.venv\Scripts\python.exe experiments/pcqm_gptrans_input_ema_100k/accept.py --records platforms/_records/kaggle/training/gptrans_g1_input_ema_v1 --package platforms/_records/kaggle/packages/gptrans_g1_input_ema_v1/release/source --output experiments/pcqm_gptrans_input_ema_100k/gpu/results/acceptance.json --close-rml
```

Then validate/rebuild RML and verify the actual candidate/reference entries for
both arm trajectory IDs. A completed platform job or source check cannot
substitute for saved-output acceptance, strict terminal evidence or Replay
admission. Fail closed if any of these stages fails; retain outputs and diagnose.
