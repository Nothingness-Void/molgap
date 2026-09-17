# Imported Phase 8 Assets

This directory is retained only because the accepted repaired-2M assets and
older experiment imports still share their original retrieval layout. It is
not a production stage and it is not a source of model recommendations.

`expansion_1m/` contains local checkpoints imported from the 1M remote run:

- `extend_1m_n997445_best.pt`: 1M SchNet checkpoint.
- `gate_2gps_expansion_1m_n997445_best.pt`: 1M dual-GPS fusion checkpoint.

They are closed candidates, not registered defaults. Their completed validation
driver and decision are preserved on the `archive` branch.

The matching 1M GPS7/GPS9 checkpoints remain in the downloaded Kaggle
external-eval model bundle at
the historical Kaggle external-evaluation bundle on the `archive` branch. The
completed common evaluation rejected global promotion, so these assets remain
outside the formal registry.

The registered repaired-2M pure-2D assets are the six files named
`phase8_repaired_2m_*` in this directory. Their registry paths and hashes are
owned by `models/README.md` and the project-freeze evidence.
