# Causal-audit infrastructure failure — job 122303028

The job failed after 2 minutes 35 seconds on its first frozen forward. The DTK
runtime's PyTorch version rejects `torch.any(dim=(1, 2))`. It produced no audit
artifact and executed no training.

Replacing the tuple-axis reduction with `flatten(1).any(dim=1)` preserves the
exact boolean equation used only to protect single-node graphs in the
off-diagonal intervention. Checkpoint, data, intervention set, inference batch,
precision, and sealed-role policy remain unchanged. One infrastructure retry
is admissible.

Terminal logs and the durable handoff marker are retained under
`platforms/_records/scnet/k1_pair_token_audit_job_122303028/`.
