"""Registration/allocation and source checks; no molecular model execution."""
import ast
import inspect
import pytest
from molgap.gptrans_author_variants import MODES, PATH_MODES, SCALED_MODES
from molgap.gptrans_author_screen import validate_arm_allocation
from molgap.pcqm_gptrans_v4 import _ema_decay, _scientific_fields
from molgap.experiment_spec import ADDONS

MODE = "degree_path_bond_mean_ema999"

def test_combination_preserves_reference_recipe():
    assert MODE in set(MODES) & PATH_MODES & SCALED_MODES
    assert (MODE, "1") in ADDONS
    assert _ema_decay(MODE) == .999
    assert _scientific_fields(MODE) == _scientific_fields("degree_scale_ema999")
    assert _ema_decay("degree_path_bond_mean") == .9999

def test_single_arm_requires_reason():
    with pytest.raises(ValueError, match="reason"):
        validate_arm_allocation({"arms": {MODE: {}}})
    assert validate_arm_allocation({"arms": {MODE: {}}, "single_arm_reason": "frozen reference"}) == (MODE,)
    assert len(validate_arm_allocation({"arms": {"degree_scale": {}, "path_bond_mean": {}}})) == 2

def test_cost_counts_allocation_not_only_used_device():
    from molgap.gptrans_author_screen import run_author_screen
    source = inspect.getsource(run_author_screen)
    ast.parse(source)
    assert '"used_gpu_count": len(modes)' in source
    assert '"allocated_device_hours": elapsed * 2 / 3600' in source
    from molgap.gptrans_author_terminal import close_author_outputs
    assert 'native["allocated_device_hours"] / len(config["arms"])' in inspect.getsource(close_author_outputs)
