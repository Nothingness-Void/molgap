# Reuse and implementation provenance

Base desktop: `a386a4a2c1ba9ce3a841fb775dec0d88dada7458`.
Retained bounded500K source: `031a890b4d604fc4614cde2da10e6e76cf2f836e`.

Reuse the existing `pcqm_500k_v4_evidence` trainer, composed family hooks,
`run_legacy_500k_pair` bootstrap and complete-epoch continuation implementation.
Bring only reviewed shared execution paths into this new question; no wholesale
merge of the old experiment. Shared source packaging, hashing, CPU release,
prospective planning and RML compiler retain their existing owners. Lift the
old local preparation/continuation into one reusable platform helper, leaving
this experiment's wrapper to declare its scientific configuration.

New scientific code is limited to a weight0 K1 mode using the existing
two-forward objective; historical weight0.1 behavior remains compatible.
The accepted desktop `k1_bn_calibration` helper supplies secondary analysis.
No replacement trainer, model class, RML schema, scheduler or scientific gate.

The unchanged legacy launch schema stages its pinned target-transform JSON as
a compatibility input. Neither K1 arm uses it: both compute the same 500K
train-only target statistics. It is not a second target normalization recipe.
