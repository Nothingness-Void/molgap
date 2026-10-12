# Product arm: initialization qualification failed

On 2026-10-13, Kaggle2 kernel 138367521/version 1 retained an ERROR parent
with one complete companion and this incomplete arm. The exact preflight log
reached `gptrans_triplet_followup.read_initial_facts` through model construction
and raised `Follow-up initial state differs from frozen aggregate parent/delta`.
The worker had already checked the uploaded initial-file SHA and written its
Spec/source/arm binding. There was no optimizer step, model forward, training
checkpoint or development prediction for this arm. This is an infrastructure
outcome, not evidence against channelwise relation composition.

The validator combined format, architecture, parent tensors, complete tensor
digest, regenerated extra tensors, parameter inventory and finiteness into one
exception. The retained stack does not identify which clause failed. The same
frozen product file passed those clauses locally with torch 2.7.1+cpu, including
deterministic mode; the complete companion recorded torch 2.11.0+cu128 remotely.
Requiring regenerated random Linear tensors to match across runtimes is a
plausible portability fault, not a proven clause-level diagnosis. Neither the
frozen tensors nor the validation requirements were modified to force a pass.

The preflight had loaded fixed training graphs and a training batch before
initial-model qualification. Record that training-role access, but no
development-role access, metric, selection, exposure trajectory or causal claim.
Startup/source constraints prohibited protected roles. No missing trace or
scientific score was reconstructed.

The full parent allocation was 12,835.850653 seconds on two T4s, or 7.131028
allocated device-hours. One half, 3.565514 allocated device-hours, is attributed
to this declared arm, including its reserved idle device after the approximately
52-second failure. This is allocation accounting, not measured GPU busy time.
The other half is retained in the independently accepted companion's ledger.

Decision: `INFRASTRUCTURE_ONLY`; no Replay-ready training claim. Close this
physical attempt without retrying either arm. A recovery would require a
separately qualified failed-arm-only release, clause-specific diagnostics and
frozen-tensor verification rather than assuming identical runtime RNG outputs.
