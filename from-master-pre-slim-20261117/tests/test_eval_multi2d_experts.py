import pytest

def test_ensemble_can_be_the_baseline(cli_module):
    validate_baseline_name = cli_module("cli.eval_multi2d_experts").validate_baseline_name
    validate_baseline_name("incumbent", {"anchor", "repair"}, {"incumbent"})


def test_unknown_baseline_is_rejected(cli_module):
    validate_baseline_name = cli_module("cli.eval_multi2d_experts").validate_baseline_name
    with pytest.raises(ValueError, match="Unknown baseline"):
        validate_baseline_name("missing", {"anchor"}, {"incumbent"})
