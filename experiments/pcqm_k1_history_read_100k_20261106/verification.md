# Local qualification and preparation — 2026-11-06 JST

Reused the existing registered scientific compiler, typed FamilyRecipe builder,
source packaging, role/data pins, sandbox, native family trainer and Kaggle
submission journal. Added only this question's addon module/catalog entries and
scientific records. No experiment-specific trainer,submitter or RML framework.

Admission initially rejected Muon+composed. The infrastructure owner supplied
reviewed desktop da1fbb4c; its equivalent commit is in this owner history. Existing
shared tests were not repeated here. A separate local input correction bound the
actual reference arm's addon+Muon recipe with existing read_recipe,
build_screen_recipe,family_arm_bindings and build_addon_declaration owners;
no training or data execution was used to construct that metadata.

Luna ran each frozen arm through the existing CLI sandbox once. The reference
arm passed. The other arm's gradients,taps and parameter-perturb/save/fresh-load
checks passed,but cross-arm step0 function equality failed. This contract
explicitly uses distinct nonzero-initialized architecture paths; it does not
require cross-arm initial-output equivalence. The original FAILED report remains.

The applicable additional check used existing
run_arm_sandbox(spec,arm,frozen=(recipe,recipe,initial_state)), with inputs taken
from this arm's immutable workflow plan. Its same_recipe report passed:
repeat construction/train-eval,owned finite gradients,executed taps and perturbed
fresh-load prediction equality. This proves single-model CPU repeatability,
not original-K1/cross-arm equivalence or native GPU qualification. Other arms
were not retested. No source,model,contract or validator was changed for this check.

For shared associative taps,only the first registration owns the tied projection
parameters in named_parameters; later taps execute but have no separate owned
parameters. NOT_APPLICABLE groups are recorded rather than independent gradients.

Remote run_screen_preflight qualifies each current model's repeat-a/repeat-b/
resumed states independently. No cross-arm step0 requirement was disabled.
Standard epoch logging,actual native/process/allocation costs and observed data
role provenance remain owned by the worker. Strict replay-ready is determined
only at terminal acceptance,with two distinct complete entries and no exclusions.

Raw reports (preserved verbatim):
- D:/molgap-artifacts/reviews/k1-four-arm-20261106/current_read-sandbox.txt SHA256 902557d244a148bd8e632e069bb264761a766bf981ab0b9ec850937523fce440
- D:/molgap-artifacts/reviews/k1-four-arm-20261106/history_read-sandbox.txt SHA256 c58ee1bce8d6fe3fe248cf5efa89ca3a898a3dc99f9d4379ea88cdc0db4d38a6
- D:/molgap-artifacts/reviews/k1-four-arm-20261106/history_read-same-recipe-sandbox.txt SHA256 5084a6ca118ba5f5fd83141171170abab20f3dededac537500d9b636dc341b78
