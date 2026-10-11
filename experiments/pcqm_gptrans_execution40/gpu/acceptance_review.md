# Execution40 independent acceptance review — 2026-10-11

While physical version3 was pending, the local saved-output adapter was reviewed
against the frozen execution protocol. The original adapter checked runtime
certificates against self-reported completion fields without independently
binding those fields to the released contract provider.

The tightened adapter derives metadata only from the existing native scientific
field helper under the declared original/optimized backend. It first compares
the local providers' normalized source SHA with their immutable version3 archive
members. It then rejects any mismatching field, including seed, precision,
optimizer, schedule, target and execution backend. No model is constructed or
executed by this metadata derivation. The native60-horizon metadata remains
unchanged; actual40 completion/exposure is separately verified, never relabeled
as a completed60-pass experiment.

Worker binding now independently matches arm, study and package. Preflight and
native partial manifests must match source and fixed-data identities, with the
frozen architecture and EMA999. The retained runtime certificate must equal its
embedded copy. Epoch timing must be finite and positive. The API and CLI require
an explicit receipt, eliminating an accidental default to version1.

The focused backend/acceptance tests passed34 checks, including rehashed runtime
certificates with wrong batch or enabled TF32 and self-consistent but unfrozen
scientific fields. Both local contract providers matched the version3 released
source. No remote payload, training recipe, threshold, model, dataset or source
dataset version changed. No training or inference occurred locally. Real outputs
and terminal RML closure were still unavailable; these tests did not constitute
training acceptance or delivery qualification.
