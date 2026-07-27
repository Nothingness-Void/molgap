# Phase 8 Additive Repair-v3 1.5M Plan

## Objective

Test whether preserving the complete original 1M distribution and appending the
500K coverage-repair set succeeds where replacing the original appended half
failed. This is a controlled pure-2D gate and does not change the production
registry.

## Dataset contract

- Base: `data/raw/phase8_expansion_1m.csv`, 1,000,000 rows.
- Additive top-up: `data/raw/phase8_repair_v2_selected_500k.csv`, 500,000 rows.
- Output: `data/raw/phase8_repair_v3_1p5m.csv`, 1,500,000 rows.
- The complete 1M base is a byte-identical prefix.
- Top-up internal CID/SMILES duplicates: 0/0.
- Cross-set CID/SMILES overlaps: 0/0.
- Output SHA-256: `6f70965b20c58a89d52fc77c77bf251286042ab4f98d5c2a8b6ae6dbd10db44a`.

Assembly evidence: `results/phase8/repair_v3_1p5m_assembly_report.json`.

## Controlled 2D run

GPS7 and GPS9 use the same 500K initialization, seed, 40-epoch schedule,
learning rate, batch size, and uniform sampling as the original 1M run. Only
the additive 500K rows differ. The dual-GPS head keeps the same architecture
and optimizer as the v1/v2 control.

SCNet jobs:

- graph build/merge: `694515`
- GPS7: `694516`, after graph success
- GPS9: `694517`, after graph success
- dual-GPS head: `694518`, after both encoders succeed

The graph job builds only the new 500K rows, then atomically merges them with
the existing 1M cache. The temporary 500K cache is removed only after the
1.5M output and contiguous source indices are validated.

## Acceptance gate

Compare the 1.5M model and original 1M model on the identical common,
OOD-1000, P8-hard, and PCQM4Mv2-valid labels with paired bootstrap intervals.
Do not allocate 1.5M 3D/SchNet/fusion compute unless the pure-2D candidate is
non-regressive on PCQM and improves at least one shared OOD/hard block.

## Future 2M preparation

Kaggle job `nothingnessvoid/active-molgap-broad-candidate-fetch-r01` prepares six durable
60K fresh-candidate rounds. It excludes the complete 1.5M table and the full
726,966-row previous candidate union. Fresh rounds are 80% balanced chemistry
and 20% capped targeted chemistry. The 360K fresh pool will be combined with
the approximately 227K unused previous candidates before a separately audited
500K selection.
