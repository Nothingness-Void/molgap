# Observed local lifecycle exercise

One existing K1 channel-gain model ran on the RTX 5060 using the original
initialization and a frozen ten-pass horizon. Source commit `18174aef` owns the
FP32 target-statistic correction and local driver. Attempt 3 uses mean
5.1759748458862305 and sample standard deviation 1.237425446510315; the owner
verified these against decoded training targets. Earlier recipes/prospectives
were retained. The original and attempt 2 both failed before formal training
(recipe identity, then FP32 statistics); neither has an invented training trace.

`platforms/local/train.py` performed preparation/native qualification, five
training passes, cooperative pause at step 80, a new-process native retained-state
qualification, the remaining five passes, mechanical acceptance, terminal
staging and RML closure. PID 1256 acknowledged 80 updates / 10240 presentations;
PID 31036 resumed the same checkpoint to 160 / 20480 / ten passes. The frozen
horizon, optimizer/scheduler/RNG and first five canonical observations survived
the restart. The pause checkpoint and its exact trace are retained in the shared
resume-bundle format at `D:/molgap-artifacts/local-k1-ten-epoch-flow/attempt_3/paused_bundle`.

The external `close_attempt.py` supplied the explicit infrastructure decision
and observed role-audit inputs and checked retained counters, process IDs and
prefix identity. It did not construct a model, training trace, terminal package,
descriptor or RML schema. Thus the common CLI exercised execution and closure;
the decision and role review remained explicit inputs, not automatic science.

The fixed manifest selects official-train rows 0:2048 for training and
100000:100512 for internal development. Both ranges fall within the 3378606
official-training rows. The role audit binds the actual source/package context
and retained prediction-row identities. Official validation, test-dev and
test-challenge remained untouched.

The first acceptance attempt exposed an unrelated historical comparison-weight
dependency in the global transaction catalog. The shared fix separates catalog
metadata validation from unrelated comparison-payload availability; schemas,
IDs, metadata links and canonical trace checks remain. Current-run artifact
hashes and ordinary strict/native acceptance remain fully checked. Only
acceptance was repeated after this fix; the passed training was not repeated.
The independent default full validation/check still verifies historical payloads
and can report their missing local bytes. No old cache or weights were hydrated.

Final outcome: **INFRASTRUCTURE_ONLY**, canonical finalization and derived rebuild
COMPLETE/VALID. Replay eligibility is false and the replay-pool delta is empty.
The observed training-process wall ledger totals 29.280667099999846 seconds;
this excludes native preflight and does not measure allocated GPU/device hours.
Scores are diagnostic outputs only; no scientific comparison, gain claim,
promotion or further training follows.

External retained reports and logs live under
`D:/molgap-artifacts/local-k1-ten-epoch-flow/attempt_3/`:
`first_process.log`, `resumed_process.log`,
`execution/local_training_report.json`,
`execution/local_acceptance_report.json` and `acceptance_metadata_mode.log`.
