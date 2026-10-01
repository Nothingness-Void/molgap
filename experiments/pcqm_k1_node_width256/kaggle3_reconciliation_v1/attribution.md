# K1 atom width256: retained-result attribution

This interpretation uses only retained predictions, canonical traces, manifests
and allocation records. Mechanical/runtime qualification, role acceptance,
terminal decision and Git routing belong to the reconciliation owner. It does
not authorize another run. Numeric evidence and artifact digests are in
`scientific_metrics.json`; `analyze_saved_predictions.py` reproduces the analysis
by calling the existing `paired_metrics` helper without inference.

On the identical finite 50,000 internal-development rows [100000,150000),
width256 has Gap MAE **0.142798333816 eV**, versus width192 **0.141294460821 eV**.
Reference-minus-candidate paired gain is **-1.503873 meV**; the 1,000-resample,
seed42 row bootstrap 95% interval is **[-2.464655,-0.525157] meV**. The candidate
improves 49.684% of individual absolute errors. It fails the frozen +3 meV point
gate, and the row interval excludes a positive gain on this retained cohort.
One training seed provides no measured training stochasticity. The result
contradicts the claimed minimum endpoint improvement for this contract; it does
not establish universal width harm across seeds, scales or datasets.

Both traces retain 40 observations at matched LR, optimizer-step and
sample-presentation coordinates, ending at 31,240 steps / 3,998,720 presentations.
Both minimum development trace values occur at epoch40; all prediction, trace,
selected-model and resume file digests match their output manifests. Each final
trace checkpoint identity equals its retained resume digest. Exact checkpoint
identities are linked in the JSON rather than duplicated here.

Width256's development advantage varies across exposure: +5.113 meV at epoch10,
-2.606 at20, +0.394 at30, -0.774 at35 and -1.504 at40. From35 to40,
development MAE falls by0.238 meV for width256 and0.968 meV for width192;
relative gain worsens by0.730 meV. Both best values at the final epoch show that
the available trace does not prove convergence. These selected observations are
descriptive; they do not calibrate an early-stop rule or justify longer training.

Final online training MAE is0.094491 eV for width256 versus0.098231 for width192.
Its semantics are pre-update, dropout-active training-cohort MAE; development
uses live-weight evaluation on another cohort. A lower online training metric
beside worse development performance is compatible with weaker generalization,
but **overfitting, underfitting, insufficient exposure and the causal mechanism
remain insufficient_evidence** without matched fixed-cohort evaluation and
training-seed evidence. No reference or checkpoint was manufactured to fill
those gaps. Post-hoc residual summaries show mean prediction-minus-target
0.005944 eV versus0.003868 eV; no calibration was fit, and this shift alone does
not explain the MAE difference. No predeclared slice evidence was retained in
this analysis.

Measured manifest invocation wall/assigned-device times are4366.358115 s for
width256 and5553.813138 s for the reference. The candidate allocation record is
explicitly a lower bound for one assigned T4 training window including
development/checkpoint, excluding bootstrap/queue and earlier resume allocations.
CPU and queue allocation are missing. These values are separate cross-job cost
observations, not architecture speedup evidence or complete two-physical-T4
allocation accounting. Both manifests identify the same Tesla T4/core
PyTorch2.10.0+cu128/CUDA12.8/cuDNN91002 and deterministic FP32 settings; installed
distribution identities differ: NumPy1.26.4 versus2.0.2, with `littleutils` and
`outdated` present only for the candidate. Runtime fingerprints therefore differ.
Controlled throughput and complete allocation comparisons are unavailable.

For the bounded endpoint question, the cheapest falsifier is already supplied
by the aligned saved predictions: width256 did not reach the material gain.
Missing causal diagnostics do not justify a repeat solely for attribution.
Any future seed, schedule or scale question requires its own decision-relevant
contract and authority; this analysis recommends no successor action.
