# Frozen G1 EMA portability decision — 2026-10-04

The exact Kaggle audit v4 passed independent saved-prediction acceptance.
Its two original 50K payloads reproduced before either disjoint development
role was opened. No optimizer, training, EMA update or checkpoint selection
was executed. Local acceptance loaded saved predictions only.

| Development cohort | EMA 0.9999 MAE | EMA 0.999 MAE | Improvement |
|---|---:|---:|---:|
| Fixed100K development, rows 100000:150000 | 0.1508848914 | 0.1442326294 | 0.0066522621 eV |
| Fixed500K development, rows 500000:550000 | 0.1555229042 | 0.1495983907 | 0.0059245135 eV |

Retention was 89.0601%, exceeding the prospective 50% gate. The paired-row
95% improvement interval on the second cohort was [0.00556509, 0.00628315]
eV, with 56.224% of molecules improved. This interval conditions on these
two fixed selected models; it does not estimate training-seed uncertainty.

## Attribution and boundary

The advantage survived changed molecules without any training update, unlike
the previously rejected small relation add-ons. This rejects cohort change
alone as an explanation for this particular EMA gain. It does not establish
that the gain survives 500K optimization or full-data training. Both weights
were selected in their original 100K training experiments; the audit did not
perform a new causal intervention. Its maximum class was PAIRED_ENDPOINT,
not STRICT_CAUSAL or a new training Replay pair.

The easiest reference-error quintile regressed by 0.01094345 eV while the other
four improved. These target-derived post-hoc groups are not deployable routes
and do not justify a specialist, conditional gate or another architecture.
The reused 500K cohort was development, not an untouched sealed test.

## Budget and decision

Measured kernel-entry-through-worker wall time was 127.1824 seconds and the
two-device allocation cost was 0.0706569 T4-hours, below the frozen audit cap.
Queue and teardown were unmeasured; this is not a Kaggle billing claim. Earlier
failed attempts retain their separate measured costs and infrastructure-only
decisions, rather than being counted as successful scientific attempts.

Outcome: NO_TRAIN positive portability, qualified for separately planned
bounded 500K EMA attribution. Preserve this terminal audit and its source
training evidence. No architecture combination, extra seed, official role,
full training or desktop custody was authorized by this audit.

The follow-up question was whether changing EMA horizon retains improvement
under matched 500K optimization, holding encoder, initialization, batch128,
FP32, optimizer, sample presentation schedule and all non-EMA fields fixed.
Release required its own prospective reference/trace/role/cost plan, executable
runtime qualification and bounded budget. The shared training capability must
be verified before a submission is claimed.

Full numeric evidence: [accepted summary](results/acceptance_summary.json).
Physical job and frozen source: [receipt](submission_receipt.json) and
[release](release_binding.json). Retrieval: [hash-bound inventory](retrieval_receipt.json).
