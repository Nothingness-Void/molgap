# Current-layer read calibration: positive below gate

2026-11-07 JST. Outcomes: original Muon CLOSED; current_read
POSITIVE_BELOW_GATE. The side path improves this same-job100K realization;
its2.376164meV gain misses the frozen3meV nomination threshold. No automatic
500K/full,model adoption,new seed or recipe search follows this decision.
Strict comparison,native qualification and replay are certified separately by
the existing acceptance owner and its canonical closure records.

| Live development observation | Original Muon | Current read |
|---|---:|---:|
| Selected MAE,eV,recomputed retained predictions | 0.132520667662 | 0.130144503669 |
| Selected pass / optimizer step | 40 / 31240 | 36 / 28116 |
| Final trace MAE,eV | 0.132520660758 | 0.130261406302 |
| Completed passes / steps | 40 / 31240 | 40 / 31240 |
| Sample presentations | 3998720 | 3998720 |
| Measured process wall,seconds | 11792.014301 | 12286.228589 |

Candidate-minus-reference selected delta=-2.376164meV;2000-resample paired-row
95% interval[-3.122626,-1.679513]meV on50000 aligned finite development rows.
Final delta=-2.259254meV;last-five matched-pass mean=-2.791223meV. All
directional tests favor the candidate; only the3meV point-effect gate fails.
The row interval is not seed variance or a probability of scale transfer.
Trace/recomputed metric differences reflect numerical reduction precision.

Process wall increases4.1911%,494.214288seconds. The shared ledger records
12607.388453wall seconds/25214.776905allocated T4-device-seconds,counted
once. Queue/provisioning before Python remain unknown. The fixed side path
adds49152parameters;this is not a free inference improvement.

## Attribution and decision value

The candidate leads at37of40 matched evaluations. Early passes1–10 and11–20
mean deltas are-3.256863 and-3.216383meV;the final10 mean is-2.632670meV.
The advantage narrows somewhat but persists through the terminal horizon.
This contradicts a purely transient early-only gain. Both traces have the
complete matched exposure; missing training cannot explain the gate miss.
More exposure improving the effect is unidentifiable from this schedule.

Final online train MAE is0.073548versus0.076275eV,candidate/reference.
These are in-pass averages under changing weights and training mode,not a
fixed-weight training evaluation; they do not establish under/overfitting.
The consistent lower development error supports useful extra global decoding,
but does not isolate attention expressivity from extra projection capacity.
That is a scientific interpretation,not a demonstrated unique mechanism.

Post-hoc median residual shifts leave-1.950547meV of the raw-2.376164meV
effect;constant calibration differences alone do not explain the gain.
The same-role equal blend scores0.127510208eV;the fitted oracle weight.58
is not adopted. These are descriptive retained-prediction statistics,not
independent validation or permission to scale an ensemble.

This calibration closes the original-Muon comparator gap left by the earlier
current/history pair. Its selected endpoints match those earlier saved results
to reduction precision; this is additional same-job control evidence for the
same seed,not independent stochastic replication. Historical accumulation's
below-gate decision is unchanged. The simpler current-only side path is a
retained positive asset;its future use requires a new explicit decision.

## Evidence

Pinned job143097149/version1,logical run
pcqm-k1-read-calibration-100k-20261107-s42-r2/v1. STATUS.md and the launch
receipt retain frozen source/Spec/package. Retrieval and diagnosis verified
complete traces/counters,selected/checkpoint artifacts and50000 finite aligned
predictions. Actual data-open provenance feeds the role audit;official
validation/test roles remain sealed. Per-arm strict/native/replay findings
are in the existing terminal closure,not inferred from these endpoint scores.

Diagnostic: `D:/molgap-artifacts/runs/pcqm-k1-read-calibration-100k-20261107-s42-r2/v1/reports/diagnostic_5c7699bb471cbcf29daf0c65b2185a06814b3abf8c8695808a901c0da5c14de6.json`.
Artifacts: matching `D:/molgap-artifacts/retrieved/` run/v1/version1.
No new training,fitted predictor or protected-role evaluation was run here.
