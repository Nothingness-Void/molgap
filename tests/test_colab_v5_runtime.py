from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "platforms" / "colab" / "v5_runtime"


def _notebook() -> dict:
    return json.loads((ADAPTER / "molgap_v5_colab_bootstrap.ipynb").read_text())


def test_colab_v5_notebook_is_gated_and_pinned() -> None:
    notebook = _notebook()
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "MOLGAP-COMMON-V5-FINAL" in source
    assert "MOLGAP-DESKTOP-V5-FINAL" in source
    assert "04cc7bfbcae5b1ae76ffaa6307a91f948e97904d" in source
    assert "STAGE_FIXED_DATA = False" in source
    assert "RUN_ACCELERATOR_PREFLIGHT = False" in source
    assert '"training_authorized": False' in source
    assert "PHYSICAL_BATCH_PER_DEVICE = 128" in source
    assert 'PRECISION = "fp32"' in source
    assert 'sys.path.insert(0, source_python)' in source
    assert "importlib.invalidate_caches()" in source


def test_colab_v5_notebook_binds_only_accepted_fixed_roles() -> None:
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in _notebook()["cells"]
    )
    assert "nvoid912/pcqm4mv2-ogb-fixed-100k-v1" in source
    assert "nvoid912/pcqm4mv2-ogb-fixed-500k-scnet-v1" in source
    assert "official_validation_role_read" in source
    assert "test_dev_role_read" in source
    assert "test_challenge_role_read" in source
    assert "official_validation" not in source.replace(
        '"official_validation_role_read"', ""
    )


def test_colab_v5_notebook_has_durable_resume_layout() -> None:
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in _notebook()["cells"]
    )
    for directory in ("contracts", "fixed_data", "runs", "checkpoints", "records"):
        assert directory in source
    assert "atomic_json" in source
    assert "checkpoint_interval" in source
    assert "resume_contract" in source
    assert "Run ID already belongs to a different immutable spec" in source
