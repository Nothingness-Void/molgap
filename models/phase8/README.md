# Phase 8 Checkpoints

`expansion_1m/` contains local checkpoints imported from the 1M remote run:

- `extend_1m_n997445_best.pt`: 1M SchNet checkpoint.
- `gate_2gps_expansion_1m_n997445_best.pt`: 1M dual-GPS fusion checkpoint.

They are closed candidates, not registered defaults. Their completed validation
driver is archived at
`scripts/phase8/archive/scaleup/validation/validate_expansion_1m_schnet.py` and
the decision is recorded under `results/phase8/expansion_1m/`.

The matching 1M GPS7/GPS9 checkpoints remain in the downloaded Kaggle
external-eval model bundle at
`results/phase8/remote/kaggle/external_eval_1m/datasets/model_assets/`. The
completed common evaluation rejected global promotion, so these assets remain
outside the formal registry.
