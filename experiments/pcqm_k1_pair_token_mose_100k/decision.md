# PairToken + MoSE31 seed-42 endpoint decision

Kaggle2 kernel `kaseichou/molgap-k1-pairtoken-mose-s42:v1` completed the
40-epoch, 31,240-step fixed PCQM-100K screen. All saved candidate artifact
hashes, source identity, checkpoint, MoSE sidecar identity, and runtime checks
passed without model inference. Official validation, test-dev, and challenge
roles were not read.

| Saved endpoint | Development Gap MAE (eV) |
|---|---:|
| K1-v4 frozen reference | 0.1413736343 |
| Original PairToken parent | 0.1383300573 |
| PairToken + MoSE31 | 0.1386557072 |

The candidate-minus-K1 paired row difference was `-0.0027179373 eV`
(`95%` row-bootstrap `[-0.0035821412, -0.0018442117]`). The candidate-minus-
PairToken difference was `+0.0003256593 eV` (`95%` row-bootstrap
`[-0.0005250969, +0.0011654410]`). Row bootstrap does not measure run-to-run
training variation. The candidate failed the prospectively stated `0.003 eV`
K1 gain and did not demonstrate a favorable parent advantage.

The frozen prelaunch incorrectly copied the K1 `feature_identity` while the
actual candidate used RWSE16+MoSE31. The shared strict screen validator
correctly rejected the resulting feature-fingerprint mismatch. Thus the run
has mechanically accepted artifacts and aligned paired endpoints, but **not**
`STRICT_CAUSAL` comparability or a valid promotion gate. This is a prelaunch
evidence-binding failure, not a reason to rewrite the frozen candidate contract
after observing the result. The local endpoint analysis is explicitly
`PAIRED_ENDPOINT`, with no causal claim attributing the difference to PairToken,
MoSE, or their interaction.

Decision: close this one extra authorized round without retry, extra seed,
500K bridge, full training, or protected-role evaluation. Retain original
PairToken as the previously qualified relation mechanism. The run is useful
for hypothesis generation only; it cannot enter the strict comparable replay
group. Numerical and hash evidence: `contextual_endpoint.json` and the
unchanged Kaggle output under
`platforms/_records/kaggle/training/pcqm_k1_pair_token_mose_s42_v1`.
