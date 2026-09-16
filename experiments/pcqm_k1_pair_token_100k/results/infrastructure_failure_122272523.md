# Infrastructure failure — job 122272523

The job ended after 2 minutes 28 seconds during architecture preflight, before
training. Runtime and fixed-cache certification passed. The sole failure was a
static parameter identity mismatch.

The contract counted the PairToken addition as 22,784 parameters but omitted
the 32 scale and 32 bias parameters of `token_norm`. The correct addition is
22,848 and the correct complete model size is 3,681,665. This correction does
not alter a tensor, initialization, data role, optimizer, schedule, exposure,
or model information flow. No scientific result exists for this attempt, so
one unchanged infrastructure retry is admissible.

Terminal evidence is retained under
`platforms/_records/scnet/k1_pair_token_job_122272523/`.
