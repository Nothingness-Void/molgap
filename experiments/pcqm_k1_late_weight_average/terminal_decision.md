# Terminal decision — 2026-10-08

NO_TRAIN. The bounded local CPU diagnostic completed. It does not qualify
training replay, independent generalization, model adoption or full release.

On the same historically selection-used50K development rows:

| Frozen output | Gap MAE, eV |
|---|---:|
| Selected epoch49, original BN |0.104904075389|
| Selected epoch49, clean BN |0.103658948278|
| Final epoch60, original BN |0.105724606524|
| Final epoch60, clean BN |0.104446280346|
| Equal endpoint parameter mean, selected BN |0.106325806108|
| Equal endpoint parameter mean, clean BN |0.103849668422|
| Equal clean endpoint prediction blend |0.103813936543|
| Contextual GPTrans EMA |0.101887690148|

The primary average-clean gain over selected-clean is **-0.190720meV**,
95% paired row CI[-0.273893,-0.110668]. The prospective1meV positive nomination
gate fails. Do not nominate this exact epoch49/60 equal parameter average.
Final-clean is worse by0.787332meV, CI[0.629833,0.950306]. Equal endpoint
prediction blending also worsens by0.154988meV.

BN calibration improves final by1.278326meV and the average by2.476138meV.
It corrects mismatched output state without making the late learned parameters
superior to selected49. Per-step EMA cannot be reconstructed from two endpoints;
this result neither validates nor rejects true EMA.

Both source identities and exact endpoint contracts were verified through the
accepted frozen factory. No buffers were averaged. All18 BN modules used the
same16384 training-feature rows with dropout disabled; original state restored.
Worker wall168.751s/process CPU598.344s; no accelerator or queue.
Detailed values and contrasts: [result](results/result.json).

Route: reviewed reusable diagnostics and evidence integrate into desktop under
the accepted-diagnostic non-promotion rule. Existing model recommendation stays
unchanged. Any true EMA or further training question needs its own prospective
contract and authorization; none is submitted by this closure.
