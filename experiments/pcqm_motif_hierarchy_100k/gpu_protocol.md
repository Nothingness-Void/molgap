# Prospective GPU screen — one motif-graph exchange on K1

The accepted [CPU partition](decision.md) qualifies only the graph sidecar.
This is a separate, seed-42, pure-2D 100K model question against the immutable
K1-v4 reference. The candidate retains K1's nine local real-bond EdgeState
blocks and three 64-channel molecular-slot exchanges. At layer 6 only, it adds
one 64-channel communication along actual directed inter-motif cut bonds:
atom mean → motif state → typed real-bond motif message → member-atom return.
The return is zero-initialized, so its initial predictions equal K1 exactly.
No coordinate, target-derived motif, teacher, second arm, or second seed is used.

This **additive** intervention supersedes the CPU protocol's *illustrative*
replacement idea. The replacement would change K1's initial function and
make an early loss hard to attribute to either deleted slot or new motif
communication. A separately frozen test would be required to study it.

The train role is fixed official-train prefix [0,100000); selection is its
next [100000,150000). The accepted fixed graph cache and sidecar hashes,
source commit, 9-category OGB atoms, 3-category OGB bonds and RWSE16 are
checked before use. Physical batch 128, seed 42, strict FP32/TF32 off,
AdamW 4e-4/1e-5, 40 epochs, 781 steps per epoch, drop-last, gradient clip 1,
cosine eta-min 1e-6, normalized Gap L1 and best internal-development MAE are
identical to K1-v4. The frozen K1 reference is reused, not retrained.

One P100 worker is requested because there is one independently justified
candidate; allocating T4x2 would leave one accelerator idle. The maximum
worker wall budget is 10 hours, with at least 15% device-memory reserve at
preflight. Failure of source, sidecar, actual model preflight, runtime tuple,
role seal, or V5 reference binding halts before scientific epochs. The runner
atomically writes last/best checkpoints each epoch and recovery archives every
10 epochs. A failed run is retained; resumption requires a reviewed new
attempt and RML trace binding, not an automatic second submission.

The scientific falsifier is the saved-prediction, row-aligned internal dev
Gap MAE versus the frozen K1-v4 0.1413736343 eV reference. Report the
paired error distribution and measured cost; a material nomination requires
at least 0.003 eV improvement under the already declared screening gate.
This reused role is not an independent generalization test. If nominated,
only a separately frozen, accepted-checkpoint, 500K internal-development
NO_TRAIN portability audit can be considered. No official validation,
test-dev/challenge, 500K fitting, full training, or automatic successor is
authorized by this screen.

The prospective RML plan and reference-bound prelaunch must pass before
publication/submission. The terminal must contain observed canonical trace,
role events, source/model/payload/checkpoint hashes, native cost and an
independently accepted decision. A source or runtime failure must not be
called a scientific loss or replay-ready completion.
