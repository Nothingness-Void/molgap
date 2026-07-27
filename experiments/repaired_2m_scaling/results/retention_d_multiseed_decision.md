# Retention-D Multi-Seed Decision

Retention-D passes the predeclared three-seed general-model gate against the
fixed retention-B control.

Across seeds 42, 43, and 44, common average MAE improves by
0.001163, 0.001083, and 0.001405 eV. OOD-1000 improves by
0.001184, 0.001167, and 0.002136 eV. P8 targeted hard improves by
0.001141, 0.000996, and 0.000658 eV. All three domains move in the preferred
direction for every seed. The mean improvements are 0.001217 eV on common,
0.001496 eV on OOD-1000, and 0.000932 eV on P8 targeted hard.

PCQM Gap regresses by 0.000502, 0.000258, and 0.002415 eV. This is stable
evidence that PCQM must remain a separately routed task domain; the accepted
PCQM GINE v4 specialist supplies that path.

Seed 43 and 44 checkpoints, training metrics, external metrics, and prediction
files were retrieved. Remote and local SHA256 values match, and all numeric
prediction fields are finite. The only stderr content is the already documented
non-fatal memory-efficient-attention warning.

Decision: accept the Retention-D training recipe as a stable general base.
Continue to use the predeclared seed-42 checkpoint for the next Oracle-only
study; seeds 43 and 44 are stability evidence rather than an automatic
three-pass deployment ensemble. Do not change the production registry or open
the future sealed 20K.

Machine-readable comparison: `retention_d_multiseed_comparison.json`.
