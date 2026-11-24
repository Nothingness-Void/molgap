# Correction note, 2026-11-24

Scope: `experiments/pcqm_gpspp_mpnn_100k_20261124/decision.md` and
`audit_closure/decision.md`. Neither file is edited: their hashes are bound by the
closure records.

- The protocol's primary metric is the selected EMA 0.999 dev MAE: delta +3.09 meV,
  row-bootstrap 95% CI +2.00 to +4.18 meV.
- The decision files call "EMA 0.9999, +0.46 meV" the pre-registered metric and call
  +3.09 meV a "point error". Both labels are wrong. EMA 0.9999 is an auxiliary readout.
- Both readouts are above the -1.4 meV close line. The outcome stays
  NEGATIVE_UNDER_CONTRACT; no number changes.
