import ast
import json
from pathlib import Path

from molgap.pcqm_k1_variants_runner import BATCH_SIZE, EPOCHS, ROWS_PER_EPOCH, SAMPLE_EXPOSURE


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/pcqm_k1_dynamic_query_100k"


def test_dynamic_query_sources_parse():
    for relative in ("accept.py", "p100_candidate/run_candidate.py"):
        path = EXPERIMENT / relative
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_dynamic_query_contract_reuses_v4_benchmark():
    contract = json.loads((EXPERIMENT / "training_contract.json").read_text(encoding="utf-8"))
    assert contract["benchmark_id"] == "pcqm4mv2-ogb-fixed-100k-gap-v4"
    assert contract["candidate"] == "neural_atom_k1_dynamic_query"
    assert contract["seed"] == 42
    assert contract["precision"] == "fp32"
    assert contract["physical_batch_per_device"] == BATCH_SIZE == 128
    assert contract["epochs"] == EPOCHS == 40
    assert contract["sample_presentations_per_epoch"] == ROWS_PER_EPOCH
    assert contract["total_sample_presentations"] == SAMPLE_EXPOSURE
    assert contract["roles"]["official_validation_role_read"] is False
    assert contract["roles"]["test_dev_role_read"] is False


def test_dynamic_query_p100_kernel_is_private_and_uses_fixed_data():
    metadata = json.loads((EXPERIMENT / "p100_candidate/kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["is_private"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaP100"
    assert "kaseichou/pcqm4mv2-ogb-fixed-100k-v1" in metadata["dataset_sources"]
