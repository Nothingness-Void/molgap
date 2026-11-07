# Decision — 2026-11-08 JST

Muon is POSITIVE_BELOW_GATE; the AdamW reference is CLOSED. The fixed recipe
retains a small development gain but fails the frozen 3 meV nomination gate.
No full-scale release, adoption, extra seed or successor training follows.

Selected saved-prediction MAE is AdamW 0.104282038302 eV (pass49, step191394)
and Muon 0.102467747769 eV (pass36, step140616). Candidate-minus-reference is
-1.814290533 meV. The predeclared 2,000 paired row resamples (seed42; aligned
50,000 development rows) give 95% [-2.385674753, -1.249630504] meV. The upper
bound is favorable, but the selected point gain is below3 meV. Row uncertainty
does not measure training-seed variance. Final and last-five matched-pass deltas
are favorable: -2.185150981 and -2.023705840 meV.

Both arms complete60 passes,234360 updates and29998080 presentations. Their
first40 trace observations match retained version2 field for field. The selected
trace reduction differs from the saved-prediction reduction by less than0.000008
meV; the saved-prediction comparison above owns the paired result.

## Attribution

The100K selected gain8.773793 meV contracts to1.814291 meV at500K under its
own matched60-pass contract. Within this500K run, selected-prefix gains shrink
from7.930338 meV at20passes to4.418053 at40 and1.814291 at60. Muon never
beats its pass36 minimum while AdamW reaches its minimum at49. Both remain
better than their earlier endpoints; the relative advantage fades with later
exposure. This supports AdamW catching up under the fixed schedule. Dataset
membership and development cohort differ between100K and500K, so the cross-
scale reduction alone does not isolate a size mechanism.

At pass60 the online normalized training metrics are0.0649920963 AdamW and
0.0538091447 Muon, lower than at40. These in-pass averages use changing
weights and differ from fixed-weight development MAE; they do not establish
underfitting, overfitting or module harm. Fixed-cohort training evaluation and
seed variability remain insufficient_evidence. EMA is tracked but the accepted
selection is Live; no EMA development curve was retained for a second claim.

Saved-error analysis finds50.774% row wins for Muon and no dominant global
bias explanation. Post-hoc physical-gap slices are favorable in all four bins,
with weaker mean gain in6–8eV (-0.501956meV); these are descriptive, not new
promotion gates. Oracle blends and same-role shift fits authorize no model.

Muon cumulative process wall80701.437882s versus AdamW36612.367646s is
2.204212x. Three physical allocations total22.703414 entry-wall hours and
45.406828 allocated T4-device-hours, including idle assigned devices; each
physical segment is counted once. All sessions remain below9h and totals below
27h/54T4h. Queue and provisioning before Python entry remain unmeasured.

Strict acceptance owns source/config/data/runtime/replay/comparison checks.
[Diagnosis](terminal_acceptance/diagnosis.json), [full curve](terminal_acceptance/curve.csv),
[allocation ledger](terminal_acceptance/allocation_ledger.json) and
[role evidence](terminal_acceptance/role_provenance/receipt.json) retain the facts.
The last resumed recorder alone cannot cover earlier invocations; exact-version
records from1/2/3 show every selected shard and no protected dataset opens.
Version1 starts without a checkpoint; versions2/3 resume. Explicit untouched
role audit states follow those records within the recorder boundary, not the
authorization. Frozen identities remain those in the original protocol and
version3 receipt. The current question needs no further training discriminator:
the complete frozen endpoint rejects nomination of this fixed recipe.
