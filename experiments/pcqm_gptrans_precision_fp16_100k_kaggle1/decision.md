# Kaggle1 GPTrans-T FP32 / FP16 precision decision

The frozen precision gate was not met. FP16 autocast completed the declared
60-epoch recipe but took 9.43% more synchronized optimizer-loop time than the
matched FP32 control (9,271.75 versus 8,472.94 T4-seconds). Median steady-state
throughput was 626.40 versus 708.29 graphs/s. The FP16 best and last development
MAE was 0.15577112 eV; FP32 was 0.15601449 eV. The paired FP16-minus-FP32 MAE
difference was -0.00024337 eV (10,000-row-bootstrap 95% interval
[-0.00120294, 0.00072650] eV). The interval crosses zero. FP16 meets the frozen
0.001 eV non-inferiority bound but fails the required 20% optimizer-time
reduction. Decision: `NEGATIVE_UNDER_CONTRACT`; no precision benefit is nominated.

Both arms report 60 epochs, 46,860 optimizer steps, and 5,998,080 presentations.
Their runtime certificates validate as Kaggle1 Tesla T4; finite prediction files
cover identical source rows 100,000–149,999. Best model, predictions, and
continuation checkpoint hashes match their completion manifests. The FP16
checkpoint retains GradScaler state. The completion, preflight, prediction
payloads, and source report official validation, test-dev, and challenge roles
sealed. Preflight peak allocated memory was 861,149,696 bytes for FP32 and
633,298,944 bytes for FP16; full-training peak GPU memory was not recorded.

Formal V5 terminal acceptance and replay readiness remain blocked. The existing
GPTrans acceptance script is scoped to pair-update normalization and its
historical external reference. The frozen Spec lacks a `prospective.same_run_replay`
binding, and the FP16 plan freezes the historical reference instead of the newly
trained same-run FP32 control. No precision-specific V5 evidence, role-use
records, native-cost records, calibrated canonical trace manifests, or terminal
acceptance metadata are emitted. Creating these by hand would invent evidence.
Consequently no `experiment_cli terminal --execute` closure or dual replay-pool
claim is recorded. The complete per-arm measurements, hashes, checks, and exact
blocker are in [results/terminal_observation.json](results/terminal_observation.json);
retained remote artifacts are under `remote_output/`.
