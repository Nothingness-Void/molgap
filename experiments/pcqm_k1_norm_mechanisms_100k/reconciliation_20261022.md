# Kaggle1 four-arm check, 2026-10-22

Source identities and artifact hashes are retained in the
[machine-readable report](reconciliation_20261022.json).
This is reconciliation and mechanical inspection, not terminal RML closure.

## Completed position contrast

Both arms reached124960 updates and15994880 gradient sample presentations.
The existing family inspector verified selected Live predictions, targets,
source rows, contract, checkpoint/resume schema and trace consistency.

| Arm | Best pass | Best MAE eV | Pass40 | Pass80 | Pass120 | Pass160 |
| --- | --- | --- | --- | --- | --- | --- |
| Post-Node-RMS | 147 | 0.1401534744 | 0.156945497 | 0.144827127 | 0.142229766 | 0.140451685 |
| Pre-Node-RMS | 135 | 0.1456912256 | 0.159871146 | 0.148682952 | 0.145903558 | 0.145834789 |

Pre minus Post best MAE is+5.537751meV; paired-row95% bootstrap interval
[+4.534687,+6.516362]meV (2000 samples, seed42). Exposure is identical and
Pre remains worse at each declared matched endpoint. Late Pre development
progress flattens; this does not justify claiming extra epochs would restore
the advantage. Another learning-rate/architecture recipe was not tested.

Fixed4096-row Live/eval training MAE at160 is0.032765295Post and0.035789322Pre;
development is0.140451685 and0.145834789. These have compatible units and
weight modes but different cohorts. Both show train/development separation,
not proof of a unique overfitting cause. This fixed training subset is not
a representative full-training estimate.

The first1000 updates clipped90.2%Post versus97.7%Pre. Both remained finite
and converged. Greater early clipping describes different optimization,
not proof that clipping explains the endpoint gap. Post-hoc scalar shift
does not remove the disadvantage (shift-corrected delta+5.870537meV).
No calibration was adopted. Row bootstrap does not measure training variance.

This supports Post over this exact Pre-RMS recipe on one seed. It does not
establish BN superiority, LN inferiority, or the value of graph-wise statistics
and adaptive gain, whose arms are incomplete.

## Incomplete scope/gain contrast

Graph-RMS retained30 observations,23430 updates,2999040 presentations;
AdaRMS retained20 observations,15620 updates,1999360 presentations. Both failed
at a10-pass checkpoint-chunk boundary. Partial scores cannot be compared as
160-pass outcomes or used to reject either mechanism.

Both errors arise in `create_resume_bundle` calling `copy_immutable_file`:
copied bytes differ from the hash captured earlier. `experiment_resume.py`
hashes sidecars before bulk copying; `kaggle_pair_runtime.py` refreshes each
arm's allocation ledger every30seconds through atomic replacement. Atomic
replacement protects JSON integrity but does not make two separate reads
refer to the same generation. This concrete race is the leading explanation;
logs omit the failing filename/hash, so exact attribution remains unproven.

The minimal repair should snapshot mutable JSON bytes once and derive both
manifest hash and retained bytes from that snapshot. Preserve strict model/
checkpoint hashing; do not bypass checksums or silently rehash changed files.
Include the failing path and expected/observed hashes in future diagnostics.
No source repair or resubmission was performed during this check.

## Retention and remaining acceptance

All retained outputs are onD:, using exact-version allowlists. About143MiB
covers two completed arms' selected models, predictions, last resume files,
traces/contracts and small metadata/logs. No cache, native preflight payload
collection or historical checkpoint chunks were downloaded. Entry-window cost
is12.084074T4-deviceh for position and2.898874T4-deviceh for scope/gain;
provisioning before entry and release afterward remain unobserved. These are
allocation units, not GPU busy time or account quota hours.

Scientific terminal decisions, complete native-runtime qualification and RML
finalization are not claimed. Scope/gain recovery requires checking retained
atomic state before any separately authorized continuation. The owning
question remains open; no scientific status was invented for missing evidence.
