# K1 slot width96: diagnostic and candidate training

Desktop-owned new question, based on integrated accepted slot/readout diagnostic.
User2026-10-02 authorizes one selected new Kaggle3 experiment, followed by local
shutdown after verified submission and saved provenance. No automatic successor.

## Intervention

Atom192, EdgeState64, one active slot,9layers, mixers3/6/9, RWSE16, original
ResGated local blocks and mean readout remain fixed. Only slot latent64->96.
Three mixers expand key/value/query, slot transformations and return projection.
Estimated194976 added parameters; qualification must measure3853793 total against
3658817 reference (about5.33% extra), below user twofold ceiling.

## Cheapest discriminator and release

Before a candidate GPU submission, prospective CPU trajectory binds code and
checkpoints. Use fixed sorted uniform2048 development offsets (NumPy RNG20261002),
CPU4threads, batch64/workers0. Strict-load retained192 selected state; reproduce
original saved predictions <=1e-4eV max deviation. Bypass only final mixer output
using input hidden, preserving earlier mixers and parameters. Measure paired
original-versus-zero-slot errors; row bootstrap1000 seed42. Capacity screen is
eligible only if removal worsens mean AE by at least0.001eV. This operational
falsifier checks utility, not proof that64 dimensions are saturated. Stop on
identity/nonfinite/reconstruction failure or300-second CPU wall ceiling.

Construct candidate seed42 CPU random state; assert default reference remains
exact accepted initial hash, finite tensors, measured parameter count and saved
roundtrip identity. Frozen initial state is published to avoid CPU/T4 constructor
differences. Candidate CPU forward on accepted real pure2D graph shape is required;
GPU repeatability, gradients, optimizer/scheduler/RNG resume and memory/runtime
qualification still occur on its assigned T4 before formal training.

## Frozen training contract

Single candidate only: V4 fixed train[0:100000], internal development[100000:150000],
seed42,40epochs,31240updates,3998720presentations, FP32/noTF32, physical batch128
drop-last, AdamW lr4e-4/wd1e-5/clip1, cosine T_max40 eta_min1e-6, normalized Gap L1,
live best-development selection. Reuse build_family_recipe and k1-screen-v1.
No baseline retraining, warm start, auxiliary objective, pretraining or ensemble.
Random initial candidate is not the frozen ablation checkpoint.

Reuse accepted private nvoid912/pcqm4mv2-ogb-fixed-100k-v1 and its byte-pinned
manifest1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d.
Existing geometry is stripped by the family loader; no geometry is constructed
or passed to the model. No dataset rebuild or role replacement.

## Reference and comparison

Retained original192 Kaggle3 reference selectedepoch40, MAE0.1412944608205557eV,
matched exposure/rows/target transform and per-arm runtime checks. Reference-only
custody imports pre-existing authority under retrospective_partial; no SSMA
verdict or old width prospective is altered. Reference identity is frozen before
new training. Candidate's sole declared scientific architecture difference is
slot width. Actual candidate runtime/software equivalence remains pending until
remote provenance; never infer full distribution equivalence from core versions.

After completion: mechanical identity/exposure/finite aligned50K predictions,
paired row bootstrap and minimum gain0.003eV. Row bootstrap excludes seed variance.
Report runtime/cost and unresolved strict qualification separately; no automatic
promotion,500K/full admission or protected-role evaluation. Actual trace and
checkpoints must be durable and hash-bound before scientific closure.

## Resource and durability

Kaggle3 explicit NvidiaTeslaT4 allocation has two physical devices; one candidate
assigneddevice0. Estimate4wall hours,8allocated T4 hours,4assigned-useful T4 hours;
unknown CPU/bootstrap/queue/full allocation measured scopes remain explicit.
Account quota and current sessions are checked independently before submission.
One existing unrelated GPTrans run is not adopted or changed.
Atomic selected/resume checkpoints, canonical trace, predictions and provenance
are written using FamilyOutputSession and retained Kaggle output. Runtime stops
fail-closed before training if preflight fails; no automatic resubmit or retry.
Local shutdown does not stop remote execution or create server monitoring.

## Offline handoff

Before shutdown, preserve immutable source/config/input/initial-state hashes,
actual kernel ID/ref/version, published source version and retrievable output
paths, push owner branch, verify authoritative remote startup status and source.
On next desktop session reconcile that exact attempt before retrieving or acting.
Official validation, test-dev, test-challenge remain untouched.
