# Reviewed bootstrap recovery

2026-10-09. Attempt1 stopped before graph/checkpoint access: package bootstrap
eagerly loaded local constants, rejected by frozen dependency identity checks.
Its original input JSON, source commit590395e1 and failed reports remain retained
under results_20261009. This was infrastructure failure, not a model result.

The parent reviewed the minimal fix: reload archived constants and refresh
package exports before importing model/cache owners; retain path/SHA enforcement.
New subprocess regressions exercise successful archive bootstrap and tampering.
The same user-authorized predeclared diagnostic receives one separately reviewed
attempt with fresh inputs_attempt2.json and results_attempt2_20261009. Scientific
procedure, original trajectories, weights, sample, roles, ceilings and gate stay
unchanged. There is no automatic retry or relaxed identity check. All attempts'
CPU/wall observations must remain visible in final cost reporting.
