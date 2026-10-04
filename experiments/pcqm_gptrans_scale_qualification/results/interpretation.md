# Interpretation of G1 scale execution qualification

On 2026-10-04, exact v1 metadata and all output-manifest hashes passed the
owning adapter. The shared terminal pipeline completed with VALID / NO_TRAIN;
no training Replay entry was added. Detailed measurements and frozen source,
runtime, fixture, observed role and native-cost bindings live in
[acceptance](acceptance.json). The original terminal decision and prospective
bytes were not rewritten by this interpretation.

## What the evidence establishes

- An isolated FP32/BS128 T4 executed the accepted G1 model and two independent
  EMA state filters against one disposable live optimizer trajectory.
- Two seeded optimizer-inclusive repeats produced identical loss and live
  state hashes. Reserved GPU memory stayed well inside the frozen headroom gate.
- The short optimizer throughput and train-role evaluation proxy met the
  frozen equal-exposure runtime estimate gate. Full two-device allocation,
  including the idle device, was recorded; the cost excludes queue/teardown.
- Only fixed500K training graphs were consumed. No development tensors,
  protected evaluation role, selected model or scientific Gap result was used.

## What it does not establish

The training estimate is for **46,860 updates / 5,998,080 presentations**,
roughly twelve passes over 500K, not sixty 500K passes. Its evaluation proxy used
training graphs and excludes real development size distribution, checkpoint I/O
and setup; it is not a scientific runtime certificate or a wall-time promise.
The measured clock is insufficient to infer 500K MAE, whether EMA999 retains its
100K gain after optimization, or whether weight decay causes scale degradation.
The decay-only factors remain analytic counterfactuals, not observed weight norms.

## Bounded decision

Execution feasibility no longer blocks designing an equal-exposure 500K EMA
attribution owner. Release still needs a real scale sampler/schedule/resume
implementation, explicit same-live-trajectory reference semantics, independently
saved filter states and predictions, complete trace/role/cost bindings and a
separate bounded scientific authorization. No long successor was submitted.
The independently authorized 100K coefficient falsifier remained under its own
monitor and contract; this profile does not select its outcome.
