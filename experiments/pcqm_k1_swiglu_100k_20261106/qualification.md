# Initial qualification preparation — before execution release

This is the original planning-only snapshot. Actual later checks and submission
are owned by [execution qualification](execution_qualification.md) and [STATUS](STATUS.md).

## Selected retained reference

Use `ev-terminal-T-pcqm-k1-message-droppath-100k-s42-baseline`.
Canonical reference bundle:
`experiments/pcqm_k1_message_droppath_100k/runs/pcqm-k1-message-droppath-100k-s42/prospective/strict_qualification/9c363e4431ff81182b2a/baseline/reference_bundle.json`.
The neighboring `qualified_terminal.json` binds it. The original baseline closes
CLOSED, has no closure blockers, and its qualification records native/strict
replay VERIFIED. Its accepted recipe is
`experiments/pcqm_k1_message_droppath_100k/generated/9d282b81d38710d97582708fa26d3a608fffa93766ff813ac82fc66bb74148dd/baseline/recipe.json`.

The bundle SHA binding and retained selected-model/prediction file hashes were
verified with the existing `research_memory.paths.verify_bound_artifact` owner.
Only file bytes were hashed: no tensor was deserialized, model constructed or
role labels parsed. Selected model SHA:
`c5a21e011c350e41d88da79f07381eb23925e3521faeb94dc4a51e09f6e451ce`;
prediction SHA:
`b94de6de43ace7ce573d3afd08ae0046b5ecc19a0edc59d754b0a41cdc414cfb`.
The native initialization is the retained structural-bias-norm100K `k1.pt`,
state identity `8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd`.
Do not initialize the candidate from the trained selected checkpoint.

The older `pcqm_k1_slot_width96/reference_binding` explicitly leaves cross-source/
runtime qualification pending; it is not substituted for this qualified baseline.

## Checks actually completed

| Check | Result / scope |
|---|---|
| Both committed RML sources and targeted code/audit search | No K1 node SwiGLU record found; ESA substitute and GPTrans FFN expansion are distinct |
| Actual K1 call path and installed GPSConv source | Node MLP index/order, nine layers,192/384 widths, SiLU and two dropout sites reviewed |
| Existing accelerated execution backend | Aggregation/pooling/edge providers do not hard-code node MLP internals; K1 calls `self.mlp(output)` |
| Catalog metadata | `k1_swiglu_ffn/1`, K1/3 composed hook registered; catalog runnable is registration only |
| Existing metadata recipe builder | `candidate_recipe.json` built; all original training fields preserved |
| AST syntax and arithmetic | New model/catalog parse;147,456 matrix weights per layer and +1,152 total delta verified |
| Reference bundle, selected-model and predictions byte hashes | Verified with existing owner; no model or data execution |

## Not executed / not claimed

No forward/backward, sandbox, checkpoint restoration, native preflight, real
parameter inventory, inference, timing, GPU profiling, training or platform call
was performed. Do not label this prepared model runtime-qualified, submit-ready,
replay-ready or positive. The family trainer/backend compatibility is source-level
evidence until actual authorized qualification succeeds.

Original baseline process wall was6044.656910s. Its two-arm allocation was
6571.668752 wall seconds /13143.337504 T4-device seconds; that shared allocation
is not a standalone baseline cost. Candidate measured native cost is unknown.
No2x measured-time bound or per-epoch speedup is asserted.

No runtime credentials, chosen account, compute budget or execution authority is
bound. No canonical prospective trajectory or remote freeze is published yet:
publish it with actual diagnostic authority before any model diagnostic. Existing
`prepare` and submit front doors run a sandbox, so they were deliberately not
called as metadata-only checks. No active GPTrans artifact/source was changed.
