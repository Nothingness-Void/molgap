import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from molgap.chemical_aux_cache import build_cache, ChemicalLabelCache
from molgap.training_reproducibility import sha256_file


def inputs(tmp_path, smiles=("CCO", "c1ccccc1"), role="train"):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps({"source_index": i, "smiles": s}) + "\n" for i, s in zip([3, 9], smiles)), encoding="utf-8")
    rh = sha256_file(rows)
    path = tmp_path / "role.json"
    path.write_text(json.dumps({"role": role, "source_indices": [3, 9], "rows_sha256": rh,
                                "dataset_identity": "synthetic", "row_identity_semantics": "fixture"}))
    return rows, rh, path, sha256_file(path)


def test_cache_join_uses_identity_not_order(tmp_path):
    args = inputs(tmp_path)
    root = tmp_path / "cache"
    report = build_cache(*args, root)
    cache = ChemicalLabelCache(root, sha256_file(root / "manifest.json"), expected_role_sha256=args[3])
    batch = SimpleNamespace(source_idx=torch.tensor([9, 3]), num_graphs=2)
    cache.attach(batch)
    np.testing.assert_array_equal(batch.chemical_descriptors.numpy(), cache.descriptors[[1, 0]])
    for ids in ([9, 10], [3, 3]):
        with pytest.raises(ValueError):
            cache.attach(SimpleNamespace(source_idx=torch.tensor(ids), num_graphs=2))
    assert report["accepted"]
    with pytest.raises(ValueError):
        ChemicalLabelCache(root, sha256_file(root / "manifest.json"), expected_role_sha256="0" * 64)
    with (root / "labels.npz").open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError):
        ChemicalLabelCache(root, sha256_file(root / "manifest.json"), expected_role_sha256=args[3])


def test_failed_label_retained_no_accepted_cache(tmp_path):
    args = inputs(tmp_path, smiles=("CCO", "invalid"))
    root = tmp_path / "cache"
    report = build_cache(*args, root)
    assert not report["accepted"] and report["failures"][0]["source_index"] == 9
    assert not (root / "labels.npz").exists()
    with pytest.raises(ValueError):
        ChemicalLabelCache(root, sha256_file(root / "manifest.json"), expected_role_sha256=args[3])


def test_protected_role_and_bad_pin_fail_before_output(tmp_path):
    args = inputs(tmp_path, role="development")
    root = tmp_path / "cache"
    with pytest.raises(ValueError):
        build_cache(*args, root)
    assert not root.exists()
    with pytest.raises(ValueError):
        build_cache(args[0], args[1], args[2], "0" * 64, root)


def test_missing_rows_not_accepted(tmp_path):
    args = inputs(tmp_path, smiles=("CCO",))
    root = tmp_path / "cache"
    with pytest.raises(ValueError):
        build_cache(*args, root)
    assert not (root / "manifest.json").exists()


def test_control_addon_has_no_cache_or_changed_loss():
    from molgap.gptrans_chemical_training import ChemicalTrainingAddon
    from molgap.gptrans_objective import GPTransObjectiveConfig
    baseline = ChemicalTrainingAddon(GPTransObjectiveConfig())
    assert baseline.scientific_fields({"loss_fingerprint": "l1"}) == {"loss_fingerprint": "l1"}
    with pytest.raises(ValueError):
        ChemicalTrainingAddon(GPTransObjectiveConfig(descriptor_weight=.1))
