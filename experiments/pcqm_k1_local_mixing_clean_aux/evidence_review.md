# Evidence review, 2026-10-01

The user authorized two narrow K1 candidates after the paper/mechanism review
in Codex thread 01a0d718-4048-7b73-8d91-679369adc752. This does not authorize
reference retraining, full training, official validation or test access.

Canonical reused reference: `pcqm-k1-v4-100k-reference-s42`, accepted original
V4 MAE 0.1413736343 eV. Its source, sampler, transform, live selection and
40-epoch exposure are retained in the imported reference bundle. Its P100
runtime certificate cannot be relabelled as a new T4 certificate.

The SSMA paper https://arxiv.org/html/2409.19414 and official implementation
https://raw.githubusercontent.com/AlmogDavid/SSMA/master/ssma.py support a
mechanism lead, not a PCQM accuracy claim. The compressed, stabilised operator
has no complete-polynomial separation guarantee. Preserve all original gated
messages; use the additional kappa4 residual only at degrees 1..4. A local
independent-sum control checks capacity and implementation only; no endpoint
gain can isolate joint mixing without a trained capacity control.

The chemical auxiliary prerequisite was independently closed on its owner
branch at 17b1987a. Its descriptor gain was 2.444253 meV and fingerprint gain
1.363584 meV; both below the frozen 3 meV material gate. The fingerprint row
interval was [0.444450,2.282788] meV. These are contextual endpoint results:
strict-reference qualification and native allocated cost were absent. Thus the
new K1 fingerprint route is low-confidence bounded screening, not an expected
promotion. Clean graph fingerprint BCE differs from the closed atom corruption
plus reconstruction recipe; it is not pretraining and does not damage inputs.

Release requires verifying that the retained K1 reference is reusable under the
actual V5 comparison checks. Accepted historical facts remain historical. Missing
checkpoint events, raw files, cost or accessible scheduler records remain missing.
