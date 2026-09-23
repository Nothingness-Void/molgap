# Round-3 frozen inference submission

Submission date: 2026-09-23. The sole SCNet Kunshan job is `122843454` on
partition `kshdtest`, requesting one Hygon DCU, eight CPUs, 24 GiB host memory,
and a four-hour maximum wall time. It moved from `PENDING (Priority)` to
`RUNNING` on `e08r3n15`. No earlier Round-3 attempt existed at submission.

The immutable audit source is commit
`5b94a8600bd512e78d882f4a785e7dc85a01dcc8`, archive SHA-256
`f4596d341dc7caad888d76344027415b6c0b69b3a230f8cc385e048334049d49`.
The remote input checkpoint, prediction payload and fixed cache manifest
independently matched the protocol's frozen SHA-256 values before submission.
The job only reads the accepted 500K internal-development shard and saves ten
atomic 5,000-row intervention chunks. It does not train or open protected roles.

The existing 30-minute Luna monitor in the already-established B conversation
was rebound to exact job `122843454`. Healthy queue/running checks are silent;
terminal evidence is handed to the existing server controller A once, after
which the monitor pauses. This submission is not a scientific result or an
authorization to retry, train a successor, or run full scale.
