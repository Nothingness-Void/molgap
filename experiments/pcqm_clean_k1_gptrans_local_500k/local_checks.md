# Local preparation checks —2026-10-15

Independent executor:gpt-6-luna/max, project Python, synthetic CPU only.
One focused254-case batch:252 passed,2 failed,60.45s. The two failures were a
new assertion helper's NumPy RNG comparison and the integrated source allowlist's
overbroad `outputs` filename exception. Corrected both; only those2 cases were
rerun,2 passed in0.64s. No broad repeated testing or scientific model execution.
The exact source exception now covers `family_training_outputs.py` only.

Coverage includes legacy recipe identity, the actual shared session/BN/EMA and
atomic outputs, Clean selection, serialized interrupted/resumed equivalence,
role/order mismatch, missing owner-state rejection and scratch state/RNG isolation.
Selected state also carries its same-step owner component for attribution.
This local evidence is not native T4 qualification or a scientific benefit.

Fresh worktree custody materialized199 pre-existing ignored RML artifacts,
5,938,599,081 bytes, by byte-copy and source/target/record SHA verification.
Two frozen initialization transports were copied from their owning accepted
release. No tensor/role decoding was used for this custody work. Existing
`research_memory check --frozen --portable` passed before prospective publication.
Remote release repeats package/import/recipe/init/mount checks and native
preflight is an all-arm barrier. Actual training step progress is required
before the user-authorized shutdown.
