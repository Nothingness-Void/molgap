"""Constant relocation parity without weights, datasets or model imports."""
from __future__ import annotations

import pickle
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from molgap import constants
from molgap.compat import legacy_constants


_ROOT = Path(__file__).resolve().parents[1]
_CONSTANTS_FILE = _ROOT / "src/molgap/constants.py"
_BASELINE = "a7bc58de"


def _frozen_source(relative_path):
    return subprocess.check_output(
        ["git", "show", f"{_BASELINE}:{relative_path}"],
        cwd=_ROOT,
    ).decode("utf-8")


@pytest.fixture(scope="module")
def frozen_constants():
    # Use the identical __file__ and inherited environment so REPO_ROOT and
    # environment-dependent inputs cannot masquerade as relocation drift.
    module = ModuleType("_frozen_molgap_constants")
    module.__file__ = str(_CONSTANTS_FILE)
    exec(compile(_frozen_source("src/molgap/constants.py"), module.__file__, "exec"), vars(module))
    return module


def test_legacy_aliases_preserve_objects_and_builtin_pickle_values():
    for name in constants._LEGACY_CONSTANTS:
        value = getattr(legacy_constants, name)
        assert getattr(constants, name) is value
        assert getattr(constants, name) is getattr(constants, name)
        assert pickle.loads(pickle.dumps(value)) == value

    for arm, name in (
        ("schnet", "PARAMS_AB_SCHNET"),
        ("visnet", "PARAMS_VISNET"),
        ("tensornet", "PARAMS_TENSORNET"),
    ):
        assert constants.AB_ENCODERS[arm]["params"] is getattr(constants, name)
    assert constants.PARAMS_AB_SCHNET == constants.PARAMS_SCHNET_300K
    assert constants.PARAMS_AB_SCHNET is not constants.PARAMS_SCHNET_300K
    assert constants.PARAMS_TENSORNET is not constants.PARAMS_TENSORNET_300K


def test_registry_still_shares_parameter_objects():
    registry = constants.MODEL_REGISTRY
    assert registry["phase6_schnet"]["params"] is constants.PARAMS_PHASE6
    assert registry["phase7_schnet_300k"]["params"] is constants.PARAMS_SCHNET_300K
    assert registry["repaired_2m_gps7"]["params"] is constants.PARAMS_GPS_2D
    assert registry["tensornet_300k"]["params"] is constants.PARAMS_TENSORNET_300K


def test_explicit_star_and_package_reexports_agree(frozen_constants):
    import molgap

    exports = {}
    exec("from molgap.constants import *", exports)
    for name in vars(frozen_constants):
        if not name.startswith("_"):
            assert exports[name] is getattr(constants, name), name
    for name in (
        "DATA_PHASE3", "DATA_PHASE6_LARGE", "GRAPHS_PHASE4",
        "MODEL_PHASE4", "PARAMS_PHASE4",
    ):
        assert getattr(molgap, name) is getattr(legacy_constants, name)
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(constants, "UNKNOWN_LEGACY_CONSTANT")


def test_constants_leaf_is_lazy_and_lightweight_in_fresh_process():
    # A synthetic package bypasses __init__'s old eager constant re-exports,
    # exercising the leaf import independently of that compatibility surface.
    script = """
import importlib
import pathlib
import sys
import types

package = types.ModuleType("molgap")
package.__path__ = [str(pathlib.Path(sys.argv[1]) / "src/molgap")]
sys.modules["molgap"] = package
constants = importlib.import_module("molgap.constants")
assert "molgap.compat.legacy_constants" not in sys.modules
assert not (constants._LEGACY_CONSTANTS & vars(constants).keys())
legacy = importlib.import_module("molgap.compat.legacy_constants")
for name in constants._LEGACY_CONSTANTS:
    assert getattr(constants, name) is getattr(legacy, name)
assert not (constants._LEGACY_CONSTANTS & vars(constants).keys())
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert "molgap.inference" not in sys.modules
assert not any("training" in name for name in sys.modules if name.startswith("molgap."))
"""
    subprocess.run(
        [sys.executable, "-B", "-I", "-c", script, str(_ROOT)],
        check=True, capture_output=True, text=True,
    )


@pytest.mark.parametrize("first_import", ["molgap", "molgap.constants", "molgap.compat.legacy_constants"])
def test_public_import_orders_remain_lightweight(first_import):
    script = """
import importlib
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "src"))
importlib.import_module(sys.argv[2])
constants = importlib.import_module("molgap.constants")
legacy = importlib.import_module("molgap.compat.legacy_constants")
assert constants.PARAMS_PHASE4 is legacy.PARAMS_PHASE4
assert "torch" not in sys.modules
assert "molgap.inference" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-B", "-I", "-c", script, str(_ROOT), first_import],
        check=True, capture_output=True, text=True,
    )
