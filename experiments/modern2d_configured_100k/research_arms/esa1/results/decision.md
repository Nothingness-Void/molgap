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
molecules. At pass20 its training MAE was0.09215013 eV versus0.08724946 eV for
the original. Canonical training metrics use normalized-Gap units; these eV
values are from the producer's explicitly converted training telemetry.
The variant fitted training less closely while modestly improving development,
consistent with a small optimization/generalization tradeoff. This observation
did not establish a causal explanation, robustness across training seeds, scale
transfer, or a family-level win against K1/GPTrans.

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
calling this a Replay-Ready scientific pair. Any canonical terminal blocker was
recorded explicitly rather than bypassed or disguised as missing training data.

The endpoint decision was INCONCLUSIVE/no nomination, not a failed execution or
proof that ESA as a family was invalid. No retry, continuation, new seed, scale,
protected-role evaluation or new candidate was released.
