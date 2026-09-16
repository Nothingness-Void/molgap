# K1 V4 explainability Stage-1 decision

Job `122250328` completed on Kunshan node `e06r4n13` in 3 minutes 42 seconds.
The in-job preflight and downloaded-artifact acceptance both passed:

- 50,000 accepted development rows and 20 frozen same-contract payloads;
- cache manifest SHA-256
  `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`;
- geometry aggregate SHA-256
  `bc83a4bd9a7fd7fd6fc6fd085d78ba3caab0e40f997d1932e0f344e701dd40c5`;
- prediction manifest SHA-256
  `f5102dd077a6a1a0b78490467dbc4a2297e27bf01c91b7b167dd598d93bd9d3c`;
- no training, model inference, official-validation read, or test-role read;
- downloaded `error_attribution.json` SHA-256
  `9dcc625016295b2ba189c4abf5c5bf094a1b151462a8cfc80ddf7c7e049218b7`,
  exactly matching the completion manifest.

## Overall finding

Frozen K1-v4 reached `0.1413736414 eV` on this reused development role. The
best existing variant was `no_attention_uniform_return` at `0.1392112223 eV`,
a paired gain of `0.0021624191 eV` with normal-approximation 95% interval
`[0.0012932499, 0.0030315882]`. It remains below the frozen `0.003 eV`
material-gain gate and cannot be promoted.

The row-wise oracle across the already-closed variants reached
`0.0487263123 eV`, but it uses labels to choose a model per row and is not a
deployable fusion result. It demonstrates error diversity only.

## Coherent failure strata

The paired map identifies two related topology-extreme regimes where removing
the one-token attention and normalizing its return repeatedly helps:

| Frozen slice | K1-v4 MAE | `no_attention_uniform_return` gain | Corroborating gain |
|---|---:|---:|---:|
| at most 12 atoms | 0.158684 | +0.004239 | edge-context uniform +0.004947 |
| at most 12 bonds | 0.163223 | +0.005032 | edge-context uniform +0.005380 |
| lowest conjugated-bond fraction | 0.193830 | +0.005035 | edge-context uniform +0.005526 |
| ring-atom fraction above 0.75 | 0.146165 | +0.004166 | no-slot attention +0.005175 |
| highest RWSE-mean quintile | 0.164105 | +0.004705 | edge-context uniform +0.004916 |

The first three slices describe small, sparse, weakly conjugated graphs; the
last two describe highly cyclic/topologically extreme graphs. The same global
exchange is therefore not equally useful across molecular topology. This is a
coherent diagnostic signal, not evidence that a gated successor will
generalize.

## Decision

Stage 1 satisfies only the first condition of the protocol's causal-audit gate.
A Stage-2 frozen-checkpoint intervention is informative: compare exchange
ablations and assignment/update statistics at layers 3, 6, and 9 on
deterministic representatives of the two topology-extreme regimes and a
matched middle stratum. Stage 2 must not train or promote a model.

No old variant is reopened, no successor architecture is released, and no
additional seed, scale-up, official role, or test role is authorized by this
result. A new architecture question requires Stage 2 to identify a causal
information-flow bottleneck tied to these strata.
