# Status

The repaired label-sealed 10K shadow cache passed independent mechanical
acceptance. Its aggregate SHA-256 is
`4a9e4361247f497f5cd9911a69e69eddb1e0fecea6e33f318ada48aa594f6a44`;
all 10,000 graphs contain only OGB atom/bond features, RWSE16, and immutable row
identity. No target label or model inference entered cache construction.

Audit version 1 failed before model construction or label access because
`hasattr` is not a valid PyG store-membership test. The unchanged-contract fix
uses `"y" in graph`; its evidence is `terminal_report_audit_v1.md`.
Version 2 then reached the first GPU operation but Kaggle assigned a P100 while
its stock Torch omitted `sm_60`; it also stopped before label access. Version 3
conditionally installs the repository's previously validated
Torch 2.7.1+cu126 P100 runtime and verifies accelerator compatibility.

Kaggle2 kernel `kaseichou/molgap-pcqm-k1-shadow-audit` version 3 completed and
passed frozen no-model acceptance. K1 reached `0.1279246956 eV`, versus
`0.1340016425 eV` for the paired Full-GPS control; the paired delta was
`-0.0060769469 eV` with bootstrap 95% interval
`[-0.0079483090, -0.0042067340] eV`. K1 also used `0.632411` times the control
inference time. The shadow role is consumed and cannot be read again.

The decision is `decision.md`. The user subsequently authorized the unchanged
paired 500K bridge on Kaggle2 under `experiments/pcqm_k1_scale500k/protocol.md`.
No full-scale training, extra seed, architecture change, or official
validation/test-dev access was authorized.
