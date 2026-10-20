# Static PPGT attribution before native capture

The2026-10-20 inspection found execution risks, not measured timing shares:

1. `PPGTAuthor.forward` builds graph views and calls `graph_forward` separately
   for each member. Physical BS128 does not mean the network's matrix operations
   process128 graphs together. One full batch invokes2048 backbone blocks and
   1024 pair-stem blocks through Python loops.
2. Every graph forward recomputes24-power RRWP features. Identity and the first
   transition are explicit; the other23 powers are small matrix products. A
   BS128 forward therefore launches2944 such products, before positional MLP,
   attention and backward. These counts describe the implementation, not a
   native profiler observation or a FLOP-to-time estimate.
3. Graph-view construction scans membership/remaps per graph. `int`, `bool`,
   category/range checks and adjacency-symmetry checks can read device results
   from Python and synchronize. Native operator trace is needed to quantify it.
4. The existing shared update also performs finite checks, clipping and atomic
   persistence. They must be measured rather than assumed negligible.

The retained native run is more informative than parameter count: its first
PPGT pass was14009.43 seconds, compared with ESA full-pass observations around
408–421 seconds later in its run. These are different models and points on their
trajectories, not a controlled speedup experiment. Saved-prediction inspection
reproduced the PPGT partial endpoint and found matching target/row identities;
it did not establish a completed scientific endpoint.

The profile keeps all arithmetic, model dimensions and FP32 unchanged. It
separates host phase/self time, GPU operators and synchronization. The captured
scratch path must match unprofiled repeat/resume states. It does not profile
the unopened Common-Dev payload, full-pass startup, steady-state throughput
or all molecular sizes, and cannot alone justify a500K run.

If small-op dispatch dominates, investigate equivalent batched execution first.
If positional work dominates, investigate an accepted deterministic CPU feature
cache. If checks/serialization dominate, review cadence without weakening native
safety or recoverability. No optimization is accepted from these hypotheses.
