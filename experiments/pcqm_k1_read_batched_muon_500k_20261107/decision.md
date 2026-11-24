# Decision — 2026-11-08 JST

Current_read is NEGATIVE_UNDER_CONTRACT; the original K1 batched-Muon reference
is CLOSED. This fixed current-only read intervention does not retain its100K
benefit at500K. No adoption, full-scale release, extra seed or successor follows.

## Complete matched comparison

Both arms finish60passes/234360updates/29998080presentations with unchanged
scientific identities. The reference selects pass54/step210924 and current_read
selects pass40/step156240. Saved-prediction development MAE is0.101863694448eV
versus0.103638072720eV: current-minus-reference +1.774378271meV. The predeclared
2000-resample paired95% interval is[+1.223561463,+2.312495555]meV (seed42,
aligned50000 rows). This fails the <=-3meV nomination threshold and upper<0
condition. The row interval is conditional on these trained models, not seed
variance. Trace-versus-saved reductions differ by less than0.000008meV.

Final MAE is0.102296173573 reference versus0.105173826218 current_read, delta
+2.877652645meV. The last-five matched-pass means are0.102351988852 versus
0.105530910194eV, delta+3.178921342meV. All three predeclared endpoint directions
are adverse. The complete60-pass [curve](terminal_acceptance/curve.csv) retains
both online training and Live development series. Original50/47-observation
prefixes match retained version1 field for field; the wall pause introduced no
scientific schedule reset.

## Attribution

The100K current-read gain2.376164meV did not transfer to this500K pair. Here
current_read never improves its pass40 selected point while the reference
improves until pass54. This supports negative benefit of this fixed read
intervention at the complete500K schedule. It does not establish that every
read architecture fails or that exposure is insufficient. The100K comparison
used unbatched Muon and another development cohort; cross-scale change alone
cannot separate membership, cohort and execution interactions.

Online normalized training means decline from pass40 to60: reference
0.0620076863 to0.0537974223; current_read0.0603118905 to0.0521644049. The lower
candidate online loss does not establish fixed-weight training performance or
overfitting: it is an in-pass average under changing weights and differs from
development MAE. Fixed-cohort train evaluation, causal failure mechanism and
training-seed variance remain insufficient_evidence. EMA was tracked, but Live
selection owns this result and no EMA development curve supports a second claim.

Saved-error analysis gives current_read48.468% row wins and finds no dominant
global bias explanation. Post-hoc physical-gap bins are all adverse, especially
<4eV (+18.721503meV;1892rows); the other bins worsen by1.034230,1.341479 and
0.304472meV. These same-development-role slices and oracle blends are descriptive
and authorize no new promotion, slice tuning or model. The completed frozen
comparison is sufficient to close this recipe; no further training is needed
merely to explain the adverse result.

## Native duration and roles

Cumulative60-pass process wall is37798.946474s (10.499707h) for original K1
with batched Muon, and40327.500599s (11.202083h) for current_read. The same-job
candidate costs1.066895x (+6.689%;2528.554125s) process wall. Dividing cumulative
process wall by60 gives10.499707 and11.202083minutes per pass, an amortized
process measure including startup/resume overhead, not isolated epoch compute.

The prior original unbatched-Muon500K job measured80701.437882s for60passes,
versus37798.946474s here: a2.135018x observed cross-job ratio (53.162% lower
process wall). This is not a strict same-job batched/unbatched speed experiment,
nor a bit-exactness claim. Both arms here use the same batched optimizer; this
scientific pair isolates current_read under that execution identity.

Physical allocation counts session1 once32223.105706s and session2 once
8790.030903s:11.392538entry-wall hours and22.785076allocated T4-device-hours,
including idle assigned devices. Both9h-session and18h/36T4h cumulative ceilings
are respected. Queue and provisioning before Python entry remain unmeasured.

[Diagnosis](terminal_acceptance/diagnosis.json) and [allocation ledger](terminal_acceptance/allocation_ledger.json)
retain numerical evidence. Both exact-version role-open records show all eleven
train/development shards plus manifest and no other dataset opens, false
protected-read flags, and matching data identity. Version1 starts without a
checkpoint; version2 resumes. The explicit [reviewed audit](terminal_acceptance/roles_reviewed/audit_map.json)
uses these [observed records](terminal_acceptance/role_provenance/receipt.json),
not authorization. Strict acceptance owns source/config/data/native/resume/replay
and causal comparison qualification; immutable source/Spec/package stay those
of the original protocol and bound version2 continuation.
