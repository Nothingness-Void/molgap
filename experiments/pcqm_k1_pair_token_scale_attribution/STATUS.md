# Status

Round-1 preflight `122742924` and the single matched60-v4 K1 500K reference
`122743291` passed terminal no-inference acceptance. Round 2 aligned saved
predictions on all 50,000 internal-development rows and established PairToken
regression against this exact K1 reference. Its dated authority is
`results/round2_decision.md`; its numerical source is
`results/round2_residual_analysis.json`.

Round 3 SCNet job `122843454` completed with exit code zero. All ten
prediction chunks passed independent no-inference acceptance, and the accepted
baseline was reproduced. The relation branch remained active, yet the exact
500K PairToken-versus-K1 comparison was unfavorable. This bounded attribution
is closed; no seed, architecture successor, full run, or protected-role access
is released. Scientific authority is `results/round3_decision.md`; scheduler,
role, cost, and artifact hashes are in `results/round3_terminal.json`.
