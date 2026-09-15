# Release decision — 2026-09-15

The accepted evidence audit identified one untested, low-cost K1 bottleneck:
the sparse global slot selector does not see the persistent bond memory that
already carries local relation information. The release changes only the
selector key with a zero-initialized incident-edge context projection.

This is a one-candidate seed-42 screen on the immutable PCQM-100K v4 contract.
It is deliberately not combined with K1-G, K1-R, geometry, path state, local
operator changes, pretraining, or GPTrans-T. A result below the material gate
closes this question; a pass nominates a shortlist only.
