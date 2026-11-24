# Runtime-stack clarification — 2026-11-06 JST

Existing retained profiler reports identify an actual numerical-stack difference:

| Field | Original K1 reference | Stabilized SwiGLU |
|---|---|---|
| Torch | 2.11.0+cu128 | 2.11.0+cu130 |
| CUDA runtime | 12.8 | 13.0 |
| Device | Tesla T4 | Tesla T4 |
| CPU threads | 1 | 1 |

Sources,read without downloading or executing models:
`D:/molgap-artifacts/retrieved/pcqm-k1-message-droppath-100k-s42/v1/experiment/baseline/profile_qualification/profile.json`
and
`D:/molgap-artifacts/retrieved/pcqm-k1-swiglu-prenorm-100k-20261106-s42/v1/version1/swiglu_prenorm/profile_qualification/profile.json`.
Their `runtime` objects contain the fields above.

This corrects the earlier statement that no specific runtime difference could
be identified. The complete environment manifests remain unavailable; these
reports establish at least one actual stack difference,not every difference
contributing to the fingerprints and not its quantitative effect on MAE.

The existing FamilyKernel execution check compares
`str(torch.__version__).split('+',1)[0]` with recipe.torch_version.
Both builds therefore pass the2.11.0 check. The submitted bootstrap did not pin
the Torch CUDA build,so this admission check did not enforce the full numerical
stack. Strict terminal comparison subsequently caught the runtime-fingerprint
mismatch. Neither identical base Torch versions nor separate repeat/resume
certificates establish equivalence across these two builds.

Future prevention should bind and validate the full Torch build/CUDA runtime
before formal training and retain the complete observed runtime manifest for
diagnosis. This is a proposed infrastructure follow-up,not a change implemented
by this addendum or permission to retrain the reference.

The accepted INCONCLUSIVE decision,original fingerprints,artifacts and RML
finalization remain byte-for-byte unchanged. No retroactive strict qualification,
new training,download,inference or repeated acceptance was performed.
