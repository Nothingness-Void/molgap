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

Remote run_screen_preflight qualifies each current model's repeat-a/repeat-b/
resumed states independently. No cross-arm step0 requirement was disabled.
Standard epoch logging,actual native/process/allocation costs and observed data
role provenance remain owned by the worker. Strict replay-ready is determined
only at terminal acceptance,with two distinct complete entries and no exclusions.

Raw reports (preserved verbatim):
- D:/molgap-artifacts/reviews/k1-four-arm-20261106/message_mean-sandbox.txt SHA256 bc3cd84107cd21dac8e0c96b0db573b17a8cc596423b16b943b35baae9a8f979
- D:/molgap-artifacts/reviews/k1-four-arm-20261106/message_std-sandbox.txt SHA256 6dc34493c58da37f0ead04d6e92ec1106664b7c92fd753f9411744ccb93aaa5a
- D:/molgap-artifacts/reviews/k1-four-arm-20261106/message_std-same-recipe-sandbox.txt SHA256 38eca8ca968cd3e70dec88f122a7be80f067c8e19288b29b08f15d038c042319
