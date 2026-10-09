# Terminal decision - 2026-10-09

NO_TRAIN. Accepted bounded frozen clean-fit description, not independent
generalization qualification, causal underfit/overfit proof or model adoption.
The original consistency500K material nomination and historical records remain
unchanged. No successor, training, protected-role evaluation or promotion follows.

MAE and dev-minus-train gaps are Gap/eV. Train is the same fixed16384 sample;
development is the full50000 selection-consumed cohort. Epoch49 development
predictions were reused after hash/state/software checks, not inferred again.

| Arm / epoch / state | Train MAE | Development MAE | Dev-train gap |
|---|---:|---:|---:|
| mean2 /49 /original |0.062179920|0.105022890|0.042842970|
| mean2 /49 /calibrated |0.057315002|0.103952010|0.046637008|
| mean2 /60 /original |0.061968722|0.105566379|0.043597657|
| mean2 /60 /calibrated |0.056045104|0.103898591|0.047853487|
| consistency /49 /original |0.061374754|0.104904075|0.043529321|
| consistency /49 /calibrated |0.056516516|0.103658948|0.047142433|
| consistency /60 /original |0.061651635|0.105724607|0.044072971|
| consistency /60 /calibrated |0.055978882|0.104446280|0.048467398|

For calibrated49->60, mean2 sampled train improves1.269898meV
(change CI[-1.538769,-1.024764]), while development changes-0.053419meV
(CI[-0.212400,0.103992]). Consistency sampled train improves0.537633meV
(CI[-0.788383,-0.287410]), while development worsens0.787332meV
(CI[0.629171,0.930985]). Positive change means increased error. Gap changes
are+1.216479/+1.324965meV; gaps are descriptive differences across disjoint roles.

At calibrated60, sampled train arm difference is-0.066221meV with CI crossing0;
development consistency-minusmean2 is+0.547690meV, CI[0.035475,1.080970].
This late descriptive contrast is below1meV point difference and is not the
original selected-state nomination gate. It supplies no causal penalty verdict.
All24 within-role paired epoch/arm/BN contrasts are retained in analysis.json.
Intervals are1000 paired-row draws/seed42, not training-seed uncertainty.

All four checkpoint identities, source inventory, finite ordered predictions,
matching parameters/BN states, target alignment and exact full-state restoration
pass. Epoch60 uses its own recalibration, never epoch49 buffers. No optimizer,
gradients or protected role was used. Numerical worker wall266.458956s and
CPU-process1015.234375s; parent-observed wall267.581591s, no timeout/retry.
Saved-analysis wall3.385996s/CPU-process4.859375s are separate from worker cost.
Historical reused-prediction execution is not charged again. Preparation/tests/
closure/Git time remains unmeasured. No accelerator/queue cost applies.

RML planning succeeded before inference. Scientific decision is NO_TRAIN;
the first metadata finalization attempt failed on inherited artifact custody:
platforms/_records/kaggle/training/pcqm_gptrans_local_inductive_bias_100k_s42_v2/attempt_002/arms/rwse16_local_edge/output/best_model.pt.
Two missing inherited GPTrans checkpoints were subsequently copied from the
parent checkout with exact-byte SHA verification; whole-repository validation
passed. The retained closure receipt owns final publication status. No validator
is bypassed, generated index
manually edited or unrelated role consumed to close this diagnostic. There is no
new training trace, strict reference declaration or replay-ready claim. The
scientific recommendation and production registry remain unchanged. Reusable
implementation and accepted evidence are eligible for reviewed desktop
non-promotion integration; no commit/push/integration is performed here.
