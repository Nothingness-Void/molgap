# Kaggle1 local inductive bias attempt 001: preflight failure

Reconciled 2026-09-27 06:49:42 UTC from the Kaggle1 CLI for kernel
`nothingnessvoid/molgap-gptrans-rwse-local-bias-paired-100k-s42-v1`:
`KernelWorkerStatus.ERROR`. The initial launch receipt had reported `QUEUED`.
The error-state output contained only `rwse16/preflight.json`, its separate
runtime certificate and runtime manifest, and the kernel log. No B preflight
file or training output was available. The downloaded preflight embeds the
runtime certificate and manifest, so the separate runtime manifest is not
retained here.

The A preflight verifies source commit
`ef5d129aae44d6c097ac0a14465938ccc2ac58cd`, source archive SHA256
`243562136cda4da09597d4cabbf65e7a820e6e62fc0597d327f4ab84fa436868`,
and graph manifest SHA256
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
Its runtime certificate is accepted on one visible Tesla T4, FP32, batch 128,
with a 1.7993443-hour estimated training time within the frozen 6-hour gate.
This is preflight evidence, not a training acceptance.

The B worker failed in `_verify_model_identity` before optimizer calibration:
its generated full initial state hashed to
`96eefae91355e36eff3566e063bee88bceae903ed8cae56e8269cc984b156cc9`
instead of the frozen
`9e2e58fa54f68c061f47637a9c406dd4c849ae9b0d49623881f63bb94655db0b`.
The paired runner then reported `Pair preflight workers failed: [0, 1]`.
It never entered the training phase. A performed only its preflight optimizer
probes on training-role data; the contracted 60-epoch training exposure is
absent for both arms. The log and artifacts do not establish a measured total
device-hour cost or any development selection event. No protected-role access
is reported by the A preflight; B has no completed role certificate.

The frozen base initial-state artifact was loaded, while the B-only random
parameters were regenerated from seed 420016 on the remote runtime. The local
preflight environment uses Windows PyTorch 2.7.1, while the retained remote
manifest reports Linux PyTorch 2.10.0. Cross-runtime random initialization is
the likely source of the mismatch; the log does not expose individual B tensor
values, so the exact random-number implementation difference is not proven. A
repair would need an immutable full B initial-state payload and new frozen source/package
identities before another release decision. The owning release decision does
not authorize a retry.

Retained raw output and status observation:

| File | SHA256 |
| --- | --- |
| `kernel_status.txt` (captured CLI status) | `1907da3ce3d5b14e0c24154a9fe8781978d96e4eb63439af498c8d0a24694c54` |
| `kernel.log` | `d0084de450eded665f05ec96e0278d1bfc0ad6658d59104c95a8aaddee59b0b5` |
| `rwse16_preflight.json` | `6df19423064fe758e61efd4ec390e54ffce31c2c5cb5356e0fe3f5ef519f27c9` |

Disposition: **infrastructure failure, no scientific comparison and no
replay-ready evidence**. The prospective RML records are retained unchanged;
the research question remains unresolved. Do not close it as
`NEGATIVE_UNDER_CONTRACT`, merge the active experiment branch, or infer missing
cost, role, model, prediction, or trace evidence.
