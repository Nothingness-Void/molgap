# CPU integration decision —2026-10-16 JST

Both registered families reached MODEL_CPU_INTEGRATED under the expressly
approved synthetic-only profile. The [CPU report](integration_cpu_report.json)
retained actual results, failed attempts, hashes and scope. The
[parameter audit](parameter_audit.json) established both main budgets without
reading development data. The existing Session, optimizer, sampler, scheduler,
atomic checkpoint, source bundler and eV evaluator were reused.

The [fidelity record](../../docs/operations/MODERN2D_FIDELITY_ACCEPTANCE.md)
accepted only the exact ESA author-reference configuration. PPGT and adapted
ESA-small retained MODEL_FIDELITY_PENDING. Native preflight was not run;
scientific training was NOT_AUTHORIZED_FOR_SCIENTIFIC_TRAINING. No model ranking
or automatic cross-scale/early-stop decision was produced.

The legacy K1 extra-GPTrans inventory failure was reproduced at baseline916e0d4c
and left as a separately tracked old defect. Worktree historical CRLF failures
did not occur in primary RML. User edits and historical evidence were preserved.

Remaining release requirements were listed, not executed, in the
[non-executable contract draft](future_training_contract_draft.md). CPU task
execution stopped at this delivery; no real-role loader or native successor
was released.
