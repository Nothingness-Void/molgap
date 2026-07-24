# MolGap 1M External Evaluation

This is the Phase 8 acceptance gate. It evaluates the full 1M dual-GPS fusion
candidate against routed-v4 on the shared OOD-1000 and targeted-hard labels.

## Inputs

The three private Kaggle datasets are packaged locally under
`results/phase8/remote/kaggle/external_eval_1m/datasets/`:

- `labels/`: shared external labels.
- `model_assets/`: v4 baseline and complete 1M candidate weights.
- `runtime_source/`: frozen MolGap source snapshot for the remote job.

## Run protocol

1. Push the three datasets, then submit this directory as the private kernel.
2. Keep `RUN_MODE = "preflight"` until CUDA, RDKit, graph construction, and all
   checkpoint loads finish.
3. Change only `RUN_MODE` to `"full"`, resubmit, and download metrics plus
   predictions to `results/phase8/remote/kaggle/external_eval_1m/`.
4. Add a decision record before changing `CURRENT_STATE.md` or the model registry.

The first P100 preflight reached graph construction but failed after Torch
reinstallation left RDKit incompatible with NumPy. `external_eval_1m.py` now
pins a compatible NumPy/RDKit pair and verifies RDKit before loading MolGap.
