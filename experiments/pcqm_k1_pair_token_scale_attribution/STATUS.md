# Status

Round-1 preflight `122742924` and the single matched60-v4 K1 500K reference
`122743291` passed terminal no-inference acceptance. Round 2 aligned saved
predictions on all 50,000 internal-development rows and established PairToken
regression against this exact K1 reference. Its dated authority is
`results/round2_decision.md`; its numerical source is
`results/round2_residual_analysis.json`.

The predeclared Round-3 gate is met. The one frozen-checkpoint PairToken
inference job is SCNet `122843454` (submitted 2026-09-23). It must reproduce
the accepted baseline prediction payload and retain chunk hashes. The existing
Luna monitor is bound to that exact ID; it may not retry, train, read protected
roles, or authorize a successor or full run. Submission provenance is in
`results/round3_submission.md`.
