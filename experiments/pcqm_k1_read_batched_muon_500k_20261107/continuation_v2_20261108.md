# Wall-limited exact continuation — 2026-11-08 JST

The original logical run `pcqm-k1-read-batched-muon-500k-20261107-s42/v1`
is incomplete. Its retained pause receipts bind muon step199044 and current_read
step183783. Existing build-resume returned RESUME_VERIFIED for both arms;
prepare-resume returned RESUME_INPUTS_PREPARED. No infrastructure or scientific
configuration changed. Original source f055c137ed27b89c4eb4bc49b5e4aeabc1c84d52,
Spec ec7c2c89fb0ba6b446d9760b8b5d9ac55118ef32404be5bc2130cd65e0f99a26 and
package8290b35ad86141d8e313ffdcb868bb9a0475f359209532c4faf1be82d67caa74
remain frozen. No repair overlay is used.

## Retained progress and budget

The shared allocation ledger records failed physical session1, from
2026-11-07T04:23:13.653880Z to13:20:16.759595Z:32223.105706193entry-wall
seconds and64446.211412386allocated T4-device seconds, counted once for the
pair. Queue and pre-Python provisioning remain measurement_missing.
Against the original64800entry-wall/129600T4-device-second ceilings, balances
are32576.894293807entry-wall and65153.788587614T4-device seconds. One physical
session remains. Session2 retains the narrower32400s entry-wall limit and180s
cleanup reserve; even its complete allocation fits the observed balance.

Partial Live development MAE is not a terminal comparison. Muon best is
0.1018800362944603eV atpass49; current_read best is0.10363807529211044eV atpass40.
At matchedpass47, muon0.10211902111768723 and current_read0.10539267212152481eV
(current-minus-reference+3.273651meV). Completed traces end atpass50 and47;
no60-pass nomination, rejection or strict terminal qualification is inferred.

## Exact continuation

Original authority.json explicitly permits a wall-limited exact-state continuation
within2sessions/18entry-wall hours/36allocated T4-hours. Its SHA256 is
5b0a37d515034d076ace8f37463eca89a5b7e82efb246f6995fb88e8c388ab6b.
The existing resume plan is [resume_to_pass60.json](resume_to_pass60.json).
It retains optimizer/scheduler/RNG/EMA/counters and original60-pass cosine;
absolute terminal bound remains234360updates/29998080presentations.
Remaining updates: muon35316,current_read50577. No additional role is released.

Bundles: `D:/molgap-artifacts/resume-validation/pcqm-k1-read-batched-muon-500k-20261107-s42/v1/20261108/<arm>/resume_bundle.json`.
Prepared: `D:/molgap-artifacts/continuations/pcqm-k1-read-batched-muon-500k-20261107-s42/v2/prepared`.
Operation journal: corresponding `v2/operations/`. Existing submit_workflow owns
source/kernel version2 publication and quota checking. A bound receipt, rather
than intended version numbers, establishes the physical continuation identity.
This is progress reconciliation; no terminal RML rebuild is performed.

## Bound physical session2

Existing submit_workflow returned SUBMISSION_BOUND at2026-11-07T17:17:05.569671Z:
`kernel143169366/version2`, same Kaggle1 kernel, source datasetversion2.
Immediate scheduler observation was RUNNING. Receipt identity:
cc45a26e44e05f896528eb81c8dead804c39527fcbad1d41fbc9e0b3c670eb70.
The first call stopped after source publication because the API still exposed
readyversion1; re-entry through the same original journal reconciled source2
and submitted exactly one successor. No source recreation or identity change.
Luna owns subsequent scheduler observations.

Before publication the adapter observed available Kaggle GPU quota36.0469569444h
and checked the requested9h. This account quota is separate from native cost.

The existing exact metadata transport also retained originalversion1 data-open
provenance, separately from the already published resume package. Retrieval record:
`D:/molgap-artifacts/retrieved/pcqm-k1-read-batched-muon-500k-20261107-s42/v1/version1/role-metadata/retrieval.json`.
Two exact files under its `experiment/<arm>/` prefix:

- muon/data_open_provenance.json SHA256163041b8ffbe0da172710d67953697fcfc054ecba8cad0c23d46bf035f711483.
- current_read/data_open_provenance.json SHA256fce7d1ced5275712a40947ab84d15b36bab05a945782a82b35b1993b021f4b6c.

These are observed version-pinned bytes for later explicit cross-segment role
audit, not a terminal scientific acceptance. The published resume payload was
not altered to add these files.
