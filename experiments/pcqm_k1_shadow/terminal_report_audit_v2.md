# PCQM K1 shadow audit v2 terminal evidence

- Kernel: `kaseichou/molgap-pcqm-k1-shadow-audit`, version `2`.
- Final Kaggle2 status: `ERROR`.
- Read-only status/output commands used `C:\Users\Adminn\Documents\molgap\.venv\Scripts\kaggle.exe` with `KAGGLE_CONFIG_DIR=C:\Users\Adminn\Desktop`.
- Retrieved evidence root: `platforms/_records/kaggle/training/pcqm_k1_shadow_audit_v2`.
- Retrieved artifact: `molgap-pcqm-k1-shadow-audit.log` only. No audit metrics, completion manifest, predictions, or acceptance payload was published.
- `experiments/pcqm_k1_shadow/accept_audit.py` was not run because the remote job is incomplete and produced no valid audit root.

## Frozen identity and contract

- Launch manifest: `experiments/pcqm_k1_shadow/launch_audit_v2.json`.
- Launch manifest SHA-256: `CFA4111DFA41E1867C2D9A80A020B779DBF9A64AB29A960A2C6A8323FEADA179`.
- Source commit: `cde95f5106b4bd27a856b21a8753ddf40fd831f2`.
- Cache aggregate SHA-256: `4a9e4361247f497f5cd9911a69e69eddb1e0fecea6e33f318ada48aa594f6a44`.
- Shadow-index SHA-256: `f68f0223dccb1c0f80e76035c79efe08b0959e4e14d5b1f7ebb5153d6cbc27bd`.
- Physical batch per device: `128`.
- Launch contract: `model_training_executed=false`, `official_validation_role_read=false`, `test_dev_role_read=false`; scientific contract unchanged by the v2 repair.

## Mechanical diagnosis

The Kaggle runtime exposed a Tesla P100 with CUDA capability `sm_60`, while the installed PyTorch build supports only `sm_70` through `sm_120`. The remote audit then failed during the first inference path at the random-walk finite-value check:

```text
Found GPU0 Tesla P100-PCIE-16GB which is of cuda capability 6.0.
The current PyTorch install supports CUDA capabilities sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.
...
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

The traceback is in `pcqm_k1_shadow_audit.py:_infer` -> `qm9_gape.py:forward_gap` -> `gps.py:_encode_state_trace`, at `torch.isfinite(random_walk_pe).all()`. This is a remote runtime/PyTorch–GPU compatibility failure. No completed audit metrics were published, so the two-model shadow MAEs, paired difference/CI, inference-time ratio, and memory headroom are all `N/A`; no scientific conclusion is possible from this terminal state.

No local model execution, training, shadow-label inspection, official validation/test-dev access, successor/audit retry, or 500K submission was performed by this monitor.

## Retrieved artifact hash

| artifact | bytes | SHA-256 |
|---|---:|---|
| `molgap-pcqm-k1-shadow-audit.log` | 8109 | `87184236AD6B6FB64EAE20C271FA967067323A4CBBEFDFA97887D1D8B76BA4DB` |
