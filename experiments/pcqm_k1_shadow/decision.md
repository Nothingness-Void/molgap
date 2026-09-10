# K1 independent shadow decision

Decision date: 2026-09-11

Kaggle2 kernel `kaseichou/molgap-pcqm-k1-shadow-audit` version 3 completed
the frozen one-time audit. The no-model acceptance recomputed the paired
statistics, verified every declared artifact and identity hash, and returned
`accepted=true`. The audit trained no model and did not read official
validation or test-dev.

| model | shadow Gap MAE (eV) | inference time (s) | parameters |
|---|---:|---:|---:|
| `full_gps` | 0.1340016425 | 2.815269 | 4,771,073 |
| `neural_atom_k1` | 0.1279246956 | 1.780406 | 3,658,817 |

The paired K1-minus-Full-GPS delta was `-0.0060769469 eV`; its 20,000-repeat
bootstrap 95% interval was
`[-0.0079483090, -0.0042067340] eV`. K1 used `0.632411` times the Full-GPS
inference time. Both arms retained more than 99.6% of device memory during the
audit, far above the frozen 15% reserve gate.

K1 therefore passed every predeclared scientific and resource gate. The
shadow role was consumed by this single audit and cannot be reused for tuning,
selection, or another candidate. The audit made the frozen paired 500K bridge
eligible for a separately authorized execution. On 2026-09-11 the user chose
Kaggle2 server-side execution; this did not authorize full-scale training,
another seed, an architecture change, or access to official
validation/test-dev.

Mechanical evidence is in `terminal_report_audit_v3.md`; the retrieved remote
payload remains under
`platforms/_records/kaggle/training/pcqm_k1_shadow_audit_v3/`.
