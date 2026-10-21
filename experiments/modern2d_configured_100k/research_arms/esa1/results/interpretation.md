# ESA residual-stream screen: complete endpoint, no nomination

On2026-10-22 JST, Kaggle2 kernel140049215/version1 completed the two authorized
20-pass arms. Exact source/receipt binding, saved artifacts and the execution
retention ledger passed independent mechanical inspection. Both workers returned0
after20 complete passes,15,620 updates and1,999,360 sample presentations; this
was not STOP_FOR_COST. No local model execution or new training was performed.

## Observed endpoint

| Arm | Encoder residual | Best Common-Dev MAE | Selected pass |
|---|---|---:|---:|
|esa_lr3|author normalized stream|0.2036746740 eV|20|
|esa_identity3|identity stream, normalized attention branch|0.2026859224 eV|20|

The identity variant was lower by0.0009887516 eV on the same50,000 development
molecules. The [independent saved-prediction analysis](analysis_report.json)
recomputed0.2026859096 versus0.2036746615 eV in float64; its paired difference
was-0.0009887519 eV. The2,000-resample fixed-row95% interval was
[-0.00232647,+0.00037623] eV and crossed zero. This was not a training-seed
uncertainty interval. The candidate won25,319/50,000 rows (50.638%), not a broad
dominant improvement. At pass20 its training MAE was0.09215013 eV versus0.08724946 eV for
the original. Canonical training metrics use normalized-Gap units; these eV
values are from the producer's explicitly converted training telemetry.
The variant fitted training less closely while modestly improving development,
consistent with a small optimization/generalization tradeoff. This observation
did not establish a causal explanation, robustness across training seeds, scale
transfer, or a family-level win against K1/GPTrans.

| Pass | Original train/dev eV | Identity train/dev eV | Identity minus original Dev eV |
|---:|---|---|---:|
|5|0.226970 /0.264214|0.230920 /0.257667|-0.006547|
|10|0.161284 /0.221958|0.166208 /0.223480|+0.001522|
|15|0.114759 /0.207970|0.118986 /0.205802|-0.002168|
|20|0.087249 /0.203675|0.092150 /0.202686|-0.000989|

The development advantage was not monotonic or consistently present. Both arms
kept improving toward the cosine endpoint, so the screen did not establish a
long-horizon plateau. Nevertheless, training fit improved much more than Dev:
the terminal train/Dev gaps were about0.1164 and0.1105 eV. That combination
supported a generalization/representation limitation under this contract,
rather than evidence that merely replacing the residual stream solved ESA.

No new0.003 eV universal gate was applied. The campaign was frozen as an initial
context-only discovery screen; it had no accepted scientific comparison reference
or measured seed-variation bound. A small endpoint difference alone was therefore
insufficient for nomination, extra seeds,500K or full training. The separate
LR1e-4 slot remained independently monitored; this slot did not determine the
two-by-two residual/LR interaction by itself.

## Execution and retention

The whole observed Python allocation was13,636.4795 seconds and27,272.9590
allocated device-seconds, about7.575822 T4-hours, within its8-hour ceiling.
This shared bootstrap measurement overlaps the root and arm ledgers; adding
them would double count. Queue, pre-Python provisioning and full billing remained
unknown. The per-arm output manifests retained process wall cost but did not
independently measure allocated device cost; missing values were not filled.

Captured four-update windows took2.3320/2.3262 seconds after eight warmups.
The main process spent about1.795/1.770 host-self seconds waiting for the loader;
forward host-self was about0.260 seconds. Capture-off training spans measured
218.319/218.428 graphs per second, with observer wrappers still installed.
Peak allocated memory was907.3 MiB and reserved memory1054 MiB. The captured
remaining execution bottleneck was input supply, not evidence of insufficient
GPU memory. Loader wait does not identify which worker-side transform, shard
access or IPC operation caused it. No GPU-utilization percentage, billing gain
or predicted speedup was inferred from these short windows.

Selected and last states, optimizer/RNG/cursor state, predictions, canonical
curves, native repeat/resume artifacts and their exact SHA256s were retained
outside Git. Compact contracts, manifests, certificates, curves, profiles and
operation identity were retained under `retained/`. The
[mechanical report](mechanical_acceptance.json) verified both arms independently;
it did not by itself grant scientific acceptance or Replay eligibility.

## RML boundary

Both original prospective trajectories had empty frozen `reference_ids` and no
`same_run_replay` binding. `campaign.json` named the intended family control,
but that prose/metadata was not an RML pair binding. It was not legal to amend
the already-executed prospective snapshots after seeing results or fabricate an
accepted reference. Full curves and result evidence were preserved without
calling this a Replay-Ready scientific pair. Standard terminal staging preserved
observed train/Dev role events, process costs and both terminal packages. The
[actual finalization attempt](rml_closure_verified.json) failed closed with
`manifest reference not frozen` for both arms. No derived views were rebuilt,
no prospective record changed and no raw trace was removed to bypass the gate.
This was a prospective configuration defect, not missing result data or a reason
to retrain these models. Further scientific releases needed a correctly frozen
pair/reference route, not another generic framework or post-hoc reference claim.

The endpoint decision was INCONCLUSIVE/no nomination, not a failed execution or
proof that ESA as a family was invalid. No retry, continuation, new seed, scale,
protected-role evaluation or new candidate was released.
