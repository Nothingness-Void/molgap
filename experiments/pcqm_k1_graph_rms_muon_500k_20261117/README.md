# K1 500K GraphRMSNorm + Batched Muon Salvage

Paired same-job Kaggle T4x2 500K experiment testing whether eliminating K1's 18 legacy BatchNorm1d layers with BN-free per-graph channel `GraphRMSNorm` rescues late-epoch scaling under batched FP32 Muon.
See [protocol](protocol.md), [status](STATUS.md), and [decision](decision.md).
