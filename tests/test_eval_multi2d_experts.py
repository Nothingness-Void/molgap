import pytest

from scripts.phase8.evaluation.eval_multi2d_experts import validate_baseline_name


def test_ensemble_can_be_the_baseline():
    validate_baseline_name("incumbent", {"anchor", "repair"}, {"incumbent"})


def test_unknown_baseline_is_rejected():
    with pytest.raises(ValueError, match="Unknown baseline"):
        validate_baseline_name("missing", {"anchor"}, {"incumbent"})
