# Terminal training decision — 2026-10-24 JST

Kaggle1 kernel140519502/version1 completed both frozen40-pass arms. Exact
source/package/Spec/data identity, output hashes, native repeat/resume proofs
and mechanical outputs were verified before this scientific decision.

| Arm | Selected LIVE MAE (eV) | Selected pass / step | Outcome |
|---|---:|---:|---|
| hop_mixed |0.14296044249|40 /31240|CLOSED|
| hop_ordered |0.14184465319|40 /31240|NEGATIVE_UNDER_CONTRACT|

Ordered-minus-mixed is-1.115789meV, paired-row95% interval
[-1.974691,-0.211694]meV (50000 aligned finite rows,2000draws,seed42).
The numerical gain has a negative upper bound but does not meet the frozen
3meV material gate. Close the ordered-shell contrast under this recipe without
adoption, retry or scale-up. This does not reject every distance-aware model,
establish either arm's absolute benefit over original K1, or estimate seed variance.

Each arm completed31240updates/3998720presentations with40 trace observations,
selected/final/resumable artifacts and all four fixed4096-TRAIN/eval receipts.
Native optimizer-step overhead relative to its own original-K1 scratch probe
was22.1369%/22.5857%, below the25% admission ceiling. These short probes do not
establish full-epoch speed or absolute scientific baseline quality.

The measured Python allocation window is5958.354509564s on two TeslaT4 devices,
11916.709019128device-seconds total (3.310197nativeT4device-hours).
Queue and pre-Python provisioning remain unknown. There was no physical continuation;
the accepted startup resume proof does not imply a mid-training remote restart.

Frozen-shell inference is a separately prospective, predeclared attribution step
after strict training closure. Its outcome cannot change this training decision
or nominate a posthoc endpoint. Strict RML/replay publication is an independent
operation and must be checked after this decision is ingested.

Sources: protocol.md; submission/terminal_observation.json; retained artifacts,
diagnostic.json and fixed_train_and_structure.json under
D:/molgap-artifacts/reconciliation/k1-hop-shell-100k/.
