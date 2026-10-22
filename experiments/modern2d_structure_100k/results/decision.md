# Structural falsifiers: retain ESA count conditioning, close PPGT conditioning

On2026-10-23 JST, exact Kaggle2 kernel140243455/version1 completed both
frozen20-pass endpoints. All four preflight/train workers returned0. No new
training, inference or protected-role access was performed during acceptance.
Saved artifacts and native evidence are recorded in [acceptance](acceptance.json).

## Frozen-row result

The comparison used the prospectively pinned within-family contextual
[references](../reference_assets.json), not old K1/GPTrans scalar scores.
All50000 source indices and target values aligned exactly. Data, initial base
states, model configuration, exposure, precision, batch, optimizer, schedule,
selection and target scaling matched their respective reference declarations.
The comparison remained CONTEXT_ONLY: this was not strict reference enrollment,
seed confirmation, a cross-family ranking or Replay-Ready admission.

| Intervention | Reference MAE | Candidate MAE | Improvement | Selected pass |
|---|---:|---:|---:|---:|
| PPGT content-conditioned pairs |0.2004566171|0.1991389534|0.0013176638|20|
| ESA atom-count readout |0.2036746615|0.1993107836|0.0043638779|19|

Units are eV. Independently recomputed float64 reductions differed slightly
from the producer's float32-reduced display values. ESA's final pass20 value
was0.1994547844; its accepted selected payload belonged to pass19, not pass20.

The2000 paired row-bootstrap draws described only fixed-row uncertainty.
PPGT's candidate-minus-reference95% interval was[-0.00252550,+0.00002546] eV;
ESA's was[-0.00558122,-0.00312950]. ESA passed this protocol's exploratory
0.003eV triage criterion; PPGT did not. Neither interval estimated training-seed
variation or qualified a causal claim.

## Attribution

PPGT's conditioner learned nonzero return weights, so a disconnected addon did
not explain the small gain. Its pass10 lead of0.00725935eV contracted to
0.00131765 at pass20; the mean last-five-pass lead was0.00111085. Exactly
50.02% of rows improved. The low-Gap group improved but the6–8eV group worsened
by0.00362058. Its early optimization advantage did not become a sufficiently
large, uniform endpoint improvement. This did not falsify all content/relation
coupling: it closed this specific shared multiplicative conditioner.

ESA's two zero-start readout projections learned nonzero weights. Its candidate
led at18 of20 retained passes and by0.00419240eV on average over the last five.
51.898% of rows improved. After separate same-role median-bias removal, the
descriptive gap still favored the candidate by about0.004166eV; a mere output
offset did not account for the gain. This was an explanatory calculation, not
a deployable calibration fitted on development labels.

However, the target<4eV group (2708 rows) worsened by0.00983505eV, while the
4–6,6–8 and>=8eV groups improved. Target bins described errors, not permissible
inference-time routing signals. Predictions did not retain atom counts or
chemical descriptors, so no claim that size-specific molecules caused the gain
was made. The result supported retaining the count-conditioning hypothesis,
not declaring the family's main expressivity/generalization problem solved.

PPGT added7168 parameters (2584897 total); ESA added512 (5015745 total).
Capture-off spans measured125.18 and1199.38 graphs/s respectively, versus
144.51 and1210.38 in the frozen historical run. Those cross-job observations
were descriptive, not a paired speed benchmark. PPGT incurred about13.4% lower
throughput for a below-gate gain; ESA's observed throughput changed by under1%.
The complete-pass steady medians were928.66s and98.14s. Profile-window peaks
were763.86 and905.64MiB, not whole-run memory maxima.

## Cost, evidence and disposition

The allocation ledger measured19482.7295 Python wall seconds (about5h25m),
and38965.4590 allocated device seconds across two T4s, below25200/50400s.
Bootstrap's overlapping final observation was19483.0849s; these were not added.
ESA finished early but its allocated device remained charged while PPGT ran.
Pre-Python provisioning, queue time and actual billing remained unobserved.

PPGT content conditioning was closed without nomination. ESA count readout was
retained as one positive exploratory lead, not a500K/full/extra-seed release.
No automatic successor was authorized or submitted. Before a scientific
confirmation, the Modern2D reference/observed-role closure route needs review;
missing evidence must not be filled by inferred access or rewritten plans.
The existing Luna heartbeat was paused after its single terminal handoff.

[Summary](summary.json) owns compact exact metrics and custody;
[attribution](attribution.json) retains the20-point paired curves and target
strata. [RML status](rml_status.json) distinguishes staging from canonical
finalization and Replay admission. No family-wide rejection or K1/GPTrans
superiority claim followed from this screen.
