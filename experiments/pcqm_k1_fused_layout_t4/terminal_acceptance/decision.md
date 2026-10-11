# Attempt001 infrastructure acceptance

2026-10-11 JST. No training, inference, new role consumption, accelerator
execution, source modification, retry or model adoption in this inspection.

## Disposition

Exact kernel `nothingnessvoid/molgap-k1-fused-layout-100k-s42-v1`, ID138066440,
response version1. [API observation](../submission/status_20261011_acceptance.json)
reports ERROR, failureMessage null. [Retained report](acceptance.json) accepts
the failed-preflight artifacts only; scientific speed/quality acceptance is
NOT_EVALUABLE_NO_FORMAL_TRAINING. This is not a negative optimization result.

Both T4 workers failed during initialization. Pair state has only a preflight
phase and training_started=false for both arms: zero formal epochs, optimizer
steps or sample presentations. No prediction, train timing, runtime qualification
certificate, family output manifest or complete40 endpoint was produced.
Speedup, both MAEs and the frozen10% speed /+0.001eV noninferiority gate remain
unknown/not evaluated, never zero.

## Failure attribution

Frozen source `fdcaee6a30496804315e5c8650d48ac1f41ffb7f` uses
`k1_screen_training._load_initial_state`: torch.load returns a container but
the loader passes the entire container into state_dict_sha256. The transported
artifact contains format, model_state and state_sha256; hashing metadata strings
as tensors raises AttributeError: 'str' object has no attribute 'detach'.
Both native logs name this same call before find_fixed_cache/load_roles or any
model construction. This distinguishes transport-interface failure from fused
optimizer, layout, training fit, convergence or scientific model degradation.

The local preparation and check-release used the shared
`v4_runtime.inspect_frozen_state_artifact`, which correctly accepts the wrapper
and verifies its tensor hash. The family loader only accepted a flat state
dictionary; its existing transport unit test uses a flat toy dictionary. Prior
185 passing tests and source-byte roundtrip did not exercise the actual packaged
artifact through the family loader. The submission preparation gap is local,
not evidence of Kaggle corrupting files or optimized weights.

CPU replay of this exact loading call reproduced both AttributeErrors without
constructing models, consuming graph roles or initializing CUDA. Shared
RunContext/package checks bind each retained runtime provenance to the frozen
Spec/arm/source/archive/recipe. Initial file hashes agree with both remote
provenances; the shared inspector verifies the frozen underlying tensors.
No implementation was repaired or remote attempt repeated during acceptance.

## Cost and custody

Retained entry observation:234.706443563 wall seconds and469.412887126
allocated T4-device seconds for two physical T4s, about0.13039247 T4-device
hours inside that window. Includes entry setup/bootstrap/preflight; it is not
train time, GPU busy time, entire platform allocation or billed quota.
Queue, allocation outside entry, CPU-hours and billing remain unknown.

The eight small outputs and their observed retrieval SHA256 values live under
`platforms/_records/kaggle/training/k1_fused_layout_100k_s42_v1/attempt_001/`.
The output API addresses the latest session and does not independently attest a
script version; retained response identity and runtime source/config bindings
align, but runtime platform_version remains null. No independent output manifest
was available, so retrieval hashes must not be presented as producer-pinned SHA.

Formal RML closure is pending: preflight failure generated no qualified family
terminal output/envelope. Existing prospective trajectories remain unchanged;
no success, strict comparison, replay-ready flag or per-arm trained trace is
fabricated. This attempt is disposed as infrastructure failure while the original
speed/quality question remains unanswered on its owning branch.

Minimum repair before any separately authorized retry: align the shared
initializer inspection and family loading contract for the actual wrapper while
keeping file/tensor pins, then regression-test the transported initial artifact
through the real loader before publication. No new architecture, data, objective
or optimizer recipe is justified by this failure.
