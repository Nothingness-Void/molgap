# Execution40 infrastructure failure — 2026-10-11

Physical kernel138147173/version1 reached ERROR before either native preflight
or training. Both logs ended at the exact prospective-byte SHA gate. The
source archive normalized JSON checkout CRLF to LF, while the Spec pinned the
original plan bytes. This was transport mismatch, not model divergence or a
scientific failure; no optimizer update or checkpoint was produced.

The [retrieval receipt](failure_retrieval_v1.json) binds startup, both-failure
summary, preflight phase and native allocation. All three logs were retained
under the ignored v1 record directory. Python-window wall13.336088seconds and
two allocated T4s produced0.00740894 native allocated-device-hours; provisioning
and exact billing outside that window were unmeasured.

The corrected release transports both plan declarations as pinned raw upload
artifacts using the existing staging owner. Both staged artifacts reproduce
their original Spec SHA exactly; the check is not relaxed or normalized away.
Model, initialization, rows, FP32, BS128, LR horizon, forty-pass budget and
delivery gates remain unchanged. The user-authorized repair proceeds as a
new physical version with preserved failure/receipt history, not a duplicate
of a queued job or a scientific successor. RML records must retain the failed
attempt's native cost when the final delivery chain is closed.
