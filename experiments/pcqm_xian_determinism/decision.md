# Terminal decision: Xi'an full-model determinism audit

The bounded Xi'an Card2 audit is negative under its frozen repeatability
contract. The latest full-model replay job `67440607` completed with scheduler
exit code `0:0` on 2026-09-15, but its accepted output reports matching initial
state, predictions, and loss with **269 of 454 parameter gradient fingerprints
different across two independent processes**. All three replays within each
process matched. The final acceptance is false and the short-training retest
flag is false. A completed worker is not a passing determinism gate.

On 2026-09-23 the exact `acceptance.json`, `first.json`, and `second.json` were
retrieved again. Their bytes match the remote `SHA256SUMS`. The earlier local
copies of `first.json` and `second.json` were truncated and were not valid
JSON; the exact remote files replace them.
The tracked checksum list retains the remote digests with relative basenames.
The original remote checksum-list SHA256 is
`80ab93d49a8cd6ca04561d13a998e82e4eacd25e6b1a4e9f96c58c15d441b792`.
The no-inference audit independently
recomputed the 269 differing tensor names and verified the scheduler identity,
frozen settings, cache/source hashes, within-process identity, and protected
role flags. See `results/initial_probe/terminal_replay_20260923.json` and
`results/initial_probe/scheduler_20260923.json`.

No PCQM labels, official validation, test-dev, or challenge role was used by
the synthetic replay. The full-model gate failed, so the protocol does not
release the two short training runs or scientific ranking on this runtime.
The audit does not identify the first divergent backward operation or prove a
specific vendor-library cause. A new runtime or diagnostic question would
require its own contract.
