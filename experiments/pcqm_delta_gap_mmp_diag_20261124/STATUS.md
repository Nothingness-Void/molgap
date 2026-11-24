# Status

- **State**: `CLOSED` (`NO_TRAIN` / `INCONCLUSIVE_DO_NOT_TRAIN`; **`0`** training arms spent from `kaggle2`'s 2-slot / 4-arm budget)
- **Trajectory**: `TB-pcqm-delta-gap-mmp-diag-20261124`
- **Branch**: `codex/exp/pcqm-delta-gap-mmp-diag-20261124`
- **Scoring Run**: Local `NVIDIA GeForce RTX 5060` (`--mode local-full`, identical frozen 4-stage pipeline over SHA-verified `s43`/`s44` `EMA@60` checkpoints and `Shards 58–61 \ fit_idx`, `N = 190,450`)
- **Result Summary**:
  - `s43`: `R_real = 13.644 [13.182, 14.145] meV` vs `R_rew = 14.611 [14.176, 15.090] meV`, excess **`-0.967 [-1.210, -0.732] meV`**
  - `s44`: `R_real = 13.664 [13.197, 14.159] meV` vs `R_rew = 14.811 [14.385, 15.279] meV`, excess **`-1.147 [-1.397, -0.909] meV`**
  - Stage 3 Balance Gate: `9/9` checks passed (`unmatched=0`, `mmp_leak=0`, `same_Murcko=0.50235 vs 0.50235`)
- **Remote Kaggle Note**: `kaseichou/molgap-deltagap-mmp-diagnostic-20261124` (`v1`, `kernel_id=146939480`, `NvidiaTeslaT4`, dataset `kaseichou/molgap-delta-gap-diag-20261124-v2`) remained `KernelWorkerStatus.QUEUED` for >2h48m (`0` GPU seconds used) before local RTX 5060 fallback closed the diagnostic.
- **Artifacts**: `D:/molgap-artifacts/diagnostics/pcqm-delta-gap-mmp-diag-20261124/local_rtx5060/` (`results.json` copied into experiment dir; Stage 1/2/3 `.npz` SHA256s verified)
