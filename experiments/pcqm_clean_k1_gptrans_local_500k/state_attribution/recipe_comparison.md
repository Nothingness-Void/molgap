# Historical versus new K1 recipe

Read-only source/contract audit,2026-10-15 JST. This is not a strict training
comparison. The historical clean0.103658948eV and new0.109049475eV changed
multiple ingredients together. Same data/architecture/seed does not mean the
same initialization, stochastic trajectory or selected weight state.

| Component | Historical composed500K | New Clean EMA500K |
|---|---|---|
| Initial weights | Trained100K/10-pass backbone, original random Gap head; tensor SHA prefix7458f5c4 | Full random seed42 initialization; SHA prefix8ef6d4ba |
| Loss | Two dropout forwards; half of each normalized L1 plus0.1 mean squared prediction difference | One normalized L1 forward; no consistency/teacher/pretraining |
| Optimizer | AdamW4e-4,wd1e-5,clip1; foreach/fused explicitly false | Same nominal constants, native AdamW; foreach/fused flags not explicitly forced |
| Schedule | Epoch cosine with denominator59; final training pass at minimum LR | Native epoch cosine,T_max60; minimum reached after final pass |
| Selection | Best Live development, subsequently cleaned by separate audit | Best per-pass Clean EMA.9999 |
| BN exposure | Two stochastic forwards per gradient update | One stochastic forward; independent128-batch clean calibration per pass |
| Data | Fixed500K train/50K dev manifest630d3004,BS128,seed42+pass shuffle,drop32,60 passes | Same declared members/order/update exposure, not equal forward/BN exposure |
| Transform | Mean/sample std over all500K train | Same accepted statistics transported; no trained weights transported |
| Model | K1 atom192/edge64/slot64/layers9,dropout.05,3658817 parameters | Same architecture; later layout/mask/pooling execution source |

Historical producer code is frozen at
[`7fd5073a`](https://github.com/Nothingness-Void/molgap/tree/7fd5073afd83879d1d769f02e3ca74cd451aa007),
archive SHA0d53f2ae5a46d51cff9730e00767fa16b80ffb27002fa50b52a67c12ac8ab65a.
The later historical clean audit explicitly used inference-source commit
[`01c561e9`](https://github.com/Nothingness-Void/molgap/tree/01c561e967cb9cca4113a154eb6064cd915a716e),
archive SHAc294a785f9d0b702f63152e4448511875f7124a830f1ad088c9acdd64203814a.
These are different identities; the selected producer must not be mislabeled as
the later inference package. See the retained
[audit bindings](../../pcqm_k1_500k_bn_calibration/inputs.json) and
[acceptance snapshot](../../pcqm_k1_500k_bn_calibration/input_authority/acceptance.snapshot.json).

New source4097e696/archivea84f0242 and recipe identities remain with the
[owning protocol](../protocol.md) and [terminal decision](../terminal_acceptance/decision.md).
Historical producer and new archives were inspected as text; this source audit
performed no model/data execution or remote action. No listed recipe difference
is assigned a numerical causal share without a corresponding intervention.
