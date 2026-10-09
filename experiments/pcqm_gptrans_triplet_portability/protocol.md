# Triplet aggregation frozen-cohort diagnostic

On 2026-10-10 the user authorized submission to Kaggle2 of the cheapest
transfer diagnostic proposed after the accepted triplet100K interpretation.
This releases NO_TRAIN inference only, not500K training or an automatic successor.

## Frozen question and evidence

Does the selected triplet aggregation EMA checkpoint retain its advantage over
the selected local-bond parent on a later molecular cohort, before any500K update?
The [100K interpretation](../pcqm_gptrans_triplet_communication_100k/gpu/results/interpretation.md)
reported a0.003049777398109436eV gain. Attention did not establish an advantage.
The [accepted bottleneck diagnostic](../pcqm_gptrans_bottleneck_audit/decision_terminal.md)
already retained the local parent's predictions on the exact later cohort.

Reuse those accepted parent predictions; do not rerun or retrain the parent.
The aggregation model is the accepted zero-based epoch38 checkpoint, not a new
initialization. Reproduce its full50000 original internal-development predictions
first (max-absolute error<=0.001eV and MAE difference<=0.0001eV). Only after that
passes, evaluate10000 sorted rows selected by NumPy RandomState42 choice without
replacement from fixed500K internal development[500000,550000). This is the
previously used research cohort, not an independent sealed confirmation.

## Execution

- Accepted byte-identical fixed100K/500K graphs; no graph/conformer rebuilding.
- Pure2D inputs, direct Gap, retained100K training target transform.
- FP32, noTF32, deterministic seed42, inference batch128, eval mode.
- No optimizer, scheduler, EMA update, gradients, model repair, threshold fitting,
  checkpoint reselection, geometry, teacher, seeds43/44 or protected-role access.
- One isolated T4 worker; if Kaggle allocates a second T4 it remains unused and
  its full allocated time is counted. No second scientific candidate is released.
- Wall cap1800seconds, allocation cap1T4-hour for up to two allocated devices.
  Prior execution suggests minutes, not a new multi-hour training campaign;
  setup and failures count, unobserved queue/provisioning remain unavailable.
- Atomic5000-row chunks, progress/model/source hashes, unchanged-state proof,
  runtime, role events, allocation ledger and retrievable output manifest.

## Interpretation and terminal evidence

Report original/later paired MAE, candidate-minus-parent interval and retained
gain fraction. Retention>=50% of the frozen100K gain and a favorable later-row
95% interval permit consideration of one separately released500K study only.
Negative or uncertain retention does not justify scale-up. Row bootstrap does
not estimate training-seed variation or establish training-scale transfer.

Reuse the existing RML prospective/terminal owners. Outcome is NO_TRAIN,
comparison PAIRED_ENDPOINT, strict_ready=false; never fabricate a training trace
or call this a training Replay pair. Keep inference evidence/cost/roles complete.
Existing Luna B monitors the actual returned identity/version silently and sends
one durable terminal/fault event to server A. No new chat or cron is created.
