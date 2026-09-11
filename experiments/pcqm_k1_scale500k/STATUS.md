# Status

The K1 independent shadow audit passed and the paired 500K scientific contract
remains eligible. Kaggle2 cache versions 1 and 2 stopped before graph
construction for infrastructure-only reasons: version 1 could not resolve the
extracted source tree; version 2 omitted RDKit from its isolated dependency
installation. Version 3 reached graph construction but exposed that the draft
random 500K/10K role was neither identical to the accepted SCNet 500K/50K role
nor faithful to the accepted 100K replacement ledger. It stopped on source row
`3003839`; no GPU training was submitted.

No GPU training was submitted and no scientific result exists. The exact v2/v3
evidence is in the terminal reports. At the user's direction, the corrected
benchmark now uses the same source-index roles as SCNet: train `0..499999` and
development `500000..549999`. A further run must retain those roles,
architecture, seed 42, FP32, physical batch 128, optimizer, schedule, 40 epochs,
and decision gate.

Cache version 4 pins the same `rdkit==2026.3.6` release recorded by the accepted
SCNet-compatible cache lineage and rejects any skipped or replaced source row.

The unlaunched FP16 AMP amendment was withdrawn by the user on 2026-09-11 to
retain direct comparability with the established FP32 screens. Both paired
arms therefore use FP32, fused AdamW, pinned non-blocking transfer, and two
loader workers per arm. Physical batch 128 and every scientific variable remain
unchanged. The `full_gps` arm is a fresh full-attention control, not the old
full-data checkpoint.
