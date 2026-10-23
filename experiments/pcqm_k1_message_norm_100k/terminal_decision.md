# MsgNorm100K terminal decision, 2026-10-23 JST

Kaggle1 kernel140391249/version1 completed both40-pass arms. Exact source,
mounts and version were reconciled with the original launch receipt. The
manifest-bound outputs and native proof dependencies were retrieved selectively.
Scientific decision is separate from strict evidence qualification; the final
qualification report and STATUS own verification status.

| Endpoint | Fixed scale | Learned scale |
|---|---:|---:|
| Selected Live development MAE, eV |0.14613317214906216|0.14476912823140622|
| Selected pass / optimizer step |40 /31240|40 /31240|
| Completed passes / presentations |40 /3998720|40 /3998720|
| Observed training-process wall seconds |5746.729415805999|5707.141640945|

Primary retained-prediction comparison: learned-minus-fixed is
-0.0013640439176559448eV. The2000-resample paired-row95% interval (seed42) is
[-0.002223285236239433,-0.0005405316925048834]eV. Rows and targets align exactly
for50000 internal-development molecules. The1.364meV point improvement fails
the frozen3meV material gate despite a favorable conditional row interval.
This interval does not measure seed-to-seed training variability.

- Fixed scientific comparator closes **CLOSED**.
- Learned candidate closes **NEGATIVE_UNDER_CONTRACT**, retaining its numerical
  improvement. No adoption,500K/full scale-up, continuation or retry is released.
- This pair compares two MsgNorm variants, not MsgNorm against original K1.
  It cannot establish an absolute architectural improvement over original K1.

Both use train rows[0,100000), development[100000,150000), and Live development
selection. Exact decoded roles, target identity, source boundaries and observed
progress support the retained per-arm role audits; official validation/test
roles remain untouched. No additional model forward was used during acceptance.

The physical entry window is6026.708396261seconds with two observed Tesla T4s:
3.348171331256111 native T4-device hours, including setup and idle assignment.
The nested execution ledger accounts for3.233080220413889 T4-device hours and
must not be added again. Queue, pre-entry allocation, post-entry release and
CPU-device cost are unmeasured. Per-arm execution-ledger assignment is
1.6165401102069445 T4-device hours; broader entry cost is reported at job level.

Raw authorities are under
`D:/molgap-artifacts/reconciliation/k1-message-norm-100k/`: scheduler/source
verification, exact metadata receipts, outputs, execution retention and
`diagnostic.json`. [Attribution](attribution.md) owns interpretation and limits.
Negative history routes to archive; accepted canonical evidence remains
discoverable on desktop without adopting the rejected model implementation.
