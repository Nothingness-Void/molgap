# Status

- Prior run `pcqm-cutlass-gt-100k-20261118-s42` (`nvoid912/molgap-cutlass-gt-100k-20261118-s42`, `kernel_id=145779324`) failed at Kaggle T4 preflight (`CUDA error: no kernel image is available for execution on the device` inside `torch.linalg.inv` MAGMA). Journal preserved untouched at `D:/molgap-artifacts/runs/pcqm-cutlass-gt-100k-20261118-s42/v1`.
- Fixed in `r2-s42` by evaluating `torch.linalg.inv(l_reg.cpu())` inside `torch.no_grad()`, and retargeted to `nothingnessvoid` (`14400s` wall limit).
- Submitted `pcqm-cutlass-gt-100k-20261118-r2-s42`: `SUBMISSION_BOUND` (`nothingnessvoid/molgap-cutlass-gt-r2-100k-20261118-s42`, `kernel_id=145785120`, `version=1`, `KernelWorkerStatus.QUEUED`).
