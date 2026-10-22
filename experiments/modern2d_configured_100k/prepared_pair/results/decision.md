# Prepared two-family screen: execution accepted, no scientific nomination

On2026-10-22 JST, exact Kaggle3 kernel140182981/version1 completed both
authorized20-pass endpoints. All preflight and training workers returned0.
Independent saved-artifact inspection returned MECHANICALLY_VERIFIED for
both arms, and the original execution-retention ledger passed validation.
No local model forward, training, inference, protected-role access or new
remote job was performed during acceptance.

## Endpoint

| Arm | Recomputed CommonDev MAE (eV) | Selected pass | Training worker elapsed |
|---|---:|---:|---:|
| PPGT mean readout | 0.2004566171 |20|16756.01s|
| ESA native |0.2036746615|20|2071.89s|

Each arm completed15620 optimizer steps and1999360 formal training sample
presentations, with50000 development rows, FP32/TF32off, physical BS128 and
seed42. Producer rounded metrics differed from independently recomputed
payload MAE only at floating-point reduction precision. Selected/last states,
predictions, row/target identity, contract and canonical traces were verified.

The allocation ledger measured17019.8657 Python wall seconds (4h43m40s),
below32400 seconds. It measured34039.7314 physical device seconds across
two T4s. Kaggle session quota and physical device-hours were distinct;
provisioning-before-Python and actual billing were not observed.

## Native execution and performance

Both serialized native repeat/resume proofs were recomputed from retained
states, not certificate booleans. ESA matched the accepted backend exactly;
PPGT satisfied the frozen model/Adam tolerance with exact cursor/schedule/RNG
metadata, but was not bitwise identical. This did not establish long-run PPGT
training equivalence. The first10 shared historical PPGT development points
differed by up to0.02837 eV despite a close two-update probe; no claim of a
numerically unchanged complete trajectory was made.

Capture-off training spans, excluding the first retained span, measured about
144.51 graphs/s for PPGT and1210.38 for ESA. The steady complete-pass medians,
including development/publication, were807.54s and97.48s. Profile-window peak
allocations were582.54MiB and905.64MiB, not whole-run peak measurements.

The four retained historical ESA development points at passes5/10/15/20 were
identical to this run. Descriptive historical throughput was lower, but the
source and concurrent family workloads differed; no strictly paired causal
speedup factor was claimed. ESA finished much earlier and its allocated T4
then remained idle while PPGT finished; this idle allocation was not free in
the physical ledger, even though Kaggle quota was session-based.

## Interpretation and disposition

The authorized question, whether the optimized implementations could pass
native qualification and finish useful100K training within9h, was answered
positively. It was not an architecture comparison against a frozen K1/GPTrans
reference. These two raw MAEs did not establish a new incumbent, transferable
500K gain, or a reason to enlarge either model. ESA's final online train/Dev
gap was0.11643 eV, versus PPGT's0.06929 eV; those descriptive gaps were not a
matched inference-on-training-set or causal explanation.

The scientific disposition remained INCONCLUSIVE/context-only. No additional
training, seed,500K/full run or protected evaluation was released. Missing
prospective candidate/reference and observed-role evidence was not repaired
retrospectively, and execution acceptance did not grant Replay-Ready.

Compact exact metrics and custody hashes are in [summary](summary.json).
Raw payloads remained in configured external retained storage; the RML staging
observation separately records any canonical closure blockers.

The actual terminal-staging call failed closed at its original source/attempt
gate: prospective labels were `ppgt_mean-v2`/`esa_native-v2` (local preparation
attempt), while the translator derived `*-v1` from actual Kaggle version1.
Source commits matched. Neither namespace was rewritten after training.
Canonical RML finalization did not execute; full facts are in
[the closure observation](rml_status.json). Execution acceptance and replay
eligibility were deliberately not conflated.
