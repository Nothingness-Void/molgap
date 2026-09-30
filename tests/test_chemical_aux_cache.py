import io
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import molgap.chemical_aux_cache as chemical_aux_cache
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
    assert report["schema"] == "molgap-chemical-cache-v1"
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
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["accepted"] is False
    assert manifest["row_count"] == 0
    assert manifest["expected_row_count"] == 2
    assert manifest["export_row_count"] == 1
    assert manifest["export_coverage_complete"] is False
    assert manifest["failure_inventory_complete"] is False
    assert manifest["failures"] == [
        {"source_index": None, "reason": "input_error:Incomplete training SMILES export"}
    ]
    assert not (root / "labels.npz").exists()


def test_control_addon_has_no_cache_or_changed_loss():
    from molgap.gptrans_chemical_training import ChemicalTrainingAddon
    from molgap.gptrans_objective import GPTransObjectiveConfig
    baseline = ChemicalTrainingAddon(GPTransObjectiveConfig())
    assert baseline.scientific_fields({"loss_fingerprint": "l1"}) == {"loss_fingerprint": "l1"}
    with pytest.raises(ValueError):
        ChemicalTrainingAddon(GPTransObjectiveConfig(descriptor_weight=.1))


def _install_fake_encoder(monkeypatch, encode):
    calls = []

    class FakeEncoder:
        def __init__(
            self,
            *,
            components=("descriptors", "fingerprints"),
            parse_policy="strict",
            descriptor_missing_policy="reject",
        ):
            self.components = tuple(
                name for name in ("descriptors", "fingerprints") if name in components
            )
            self.parse_policy = parse_policy
            self.descriptor_missing_policy = descriptor_missing_policy
            names = [f"descriptor_{index}" for index in range(200)]
            names[117] = "qed"
            self.descriptor_names = (
                tuple(names) if "descriptors" in self.components else ()
            )

        def identity(self):
            return {
                "descriptor_names": list(self.descriptor_names),
                "components": list(self.components),
                "parse_policy": self.parse_policy,
                "descriptor_missing_policy": self.descriptor_missing_policy,
            }

        def encode(self, source_index, smiles):
            calls.append(source_index)
            return encode(source_index, smiles, self.components)

    monkeypatch.setattr(chemical_aux_cache, "ChemicalLabelEncoder", FakeEncoder)
    return calls


def _ok_label(source_index, components, *, descriptor_valid_mask=None):
    descriptors = None
    fingerprints = None
    if "descriptors" in components:
        descriptors = np.full(200, source_index, dtype=np.float32)
        if descriptor_valid_mask is not None:
            descriptors[~descriptor_valid_mask] = 0
    if "fingerprints" in components:
        fingerprints = np.full(512, int(source_index == 9), dtype=np.uint8)
    return SimpleNamespace(
        status="ok",
        descriptors=descriptors,
        fingerprint=fingerprints,
        descriptor_valid_mask=descriptor_valid_mask,
    )


def test_fail_fast_retains_first_failure_and_marks_partial_label_coverage(
    monkeypatch, tmp_path
):
    args = inputs(tmp_path)

    def encode(source_index, _smiles, components):
        if source_index == 3:
            return SimpleNamespace(
                status="descriptor_failure",
                descriptors=None,
                fingerprint=None,
                descriptor_valid_mask=None,
            )
        raise AssertionError("fail_fast must not encode later role rows")

    calls = _install_fake_encoder(monkeypatch, encode)
    root = tmp_path / "cache"
    report = build_cache(*args, root, fail_fast=True)

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert calls == [3]
    assert report["accepted"] is False
    assert manifest["row_count"] == 1
    assert manifest["expected_row_count"] == 2
    assert manifest["export_row_count"] == 2
    assert manifest["export_coverage_complete"] is True
    assert manifest["failure_inventory_complete"] is False
    assert manifest["failures"] == [
        {"source_index": 3, "reason": "descriptor_failure"}
    ]
    assert not (root / "labels.npz").exists()


def test_priority_rows_are_processed_first_but_saved_in_role_order(
    monkeypatch, tmp_path
):
    args = inputs(tmp_path)
    calls = _install_fake_encoder(
        monkeypatch,
        lambda source_index, _smiles, components: _ok_label(
            source_index, components
        ),
    )
    root = tmp_path / "cache"
    report = build_cache(*args, root, priority_source_indices=(9,))

    assert report["accepted"] is True
    assert calls == [9, 3]
    with np.load(root / "labels.npz", allow_pickle=False) as arrays:
        assert arrays["source_indices"].tolist() == [3, 9]
        assert arrays["descriptors"][:, 0].tolist() == [3.0, 9.0]
        assert arrays["fingerprints"][:, 0].tolist() == [0, 1]


def test_component_subset_load_attaches_only_present_arrays_and_addon_rejects_missing(
    monkeypatch, tmp_path
):
    args = inputs(tmp_path)
    _install_fake_encoder(
        monkeypatch,
        lambda source_index, _smiles, components: _ok_label(
            source_index, components
        ),
    )
    root = tmp_path / "fingerprint_cache"
    report = build_cache(*args, root, components=("fingerprints",))

    assert report["schema"] == "molgap-chemical-cache-v2"
    cache = ChemicalLabelCache(
        root, sha256_file(root / "manifest.json"), expected_role_sha256=args[3]
    )
    batch = SimpleNamespace(source_idx=torch.tensor([9, 3]), num_graphs=2)
    cache.attach(batch)
    assert cache.components == ("fingerprints",)
    assert cache.descriptors is None
    assert not hasattr(batch, "chemical_descriptors")
    assert batch.chemical_fingerprint.shape == (2, 512)

    from molgap.gptrans_chemical_training import ChemicalTrainingAddon
    from molgap.gptrans_objective import GPTransObjectiveConfig

    with pytest.raises(ValueError, match="lacks components"):
        ChemicalTrainingAddon(
            GPTransObjectiveConfig(descriptor_weight=0.1), cache=cache
        )


def test_masked_descriptor_cache_records_missing_qed_and_rejects_missing_mask(
    monkeypatch, tmp_path
):
    args = inputs(tmp_path)

    def encode(source_index, _smiles, components):
        mask = np.ones(200, dtype=np.bool_)
        if source_index == 9:
            mask[117] = False
        return _ok_label(
            source_index, components, descriptor_valid_mask=mask
        )

    _install_fake_encoder(monkeypatch, encode)
    root = tmp_path / "masked_cache"
    report = build_cache(
        *args,
        root,
        components=("descriptors",),
        descriptor_missing_policy="mask_nonfinite",
    )
    assert report["schema"] == "molgap-chemical-cache-v2"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["missing_descriptors"] == [
        {"source_index": 9, "column_indices": [117], "column_names": ["qed"]}
    ]

    cache = ChemicalLabelCache(
        root, sha256_file(manifest_path), expected_role_sha256=args[3]
    )
    assert cache.descriptor_valid_mask.shape == (2, 200)
    assert cache.descriptor_valid_mask[0].all()
    assert not cache.descriptor_valid_mask[1, 117]
    assert cache.descriptors[1, 117] == 0

    arrays_path = root / "labels.npz"
    with np.load(arrays_path, allow_pickle=False) as arrays:
        rewritten = {
            name: arrays[name].copy()
            for name in arrays.files
            if name != "descriptor_valid_mask"
        }
    buffer = io.BytesIO()
    np.savez(buffer, **rewritten)
    arrays_path.write_bytes(buffer.getvalue())
    manifest["labels_sha256"] = sha256_file(arrays_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="arrays differ"):
        ChemicalLabelCache(
            root, sha256_file(manifest_path), expected_role_sha256=args[3]
        )


def test_cache_loader_rejects_forged_label_policy_after_manifest_repin(
    monkeypatch, tmp_path
):
    args = inputs(tmp_path)
    _install_fake_encoder(
        monkeypatch,
        lambda source_index, _smiles, components: _ok_label(
            source_index, components
        ),
    )
    root = tmp_path / "policy_cache"
    build_cache(*args, root, components=("fingerprints",))

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["label_policy"]["parse_policy"] = "forged-policy"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported auxiliary label policy"):
        ChemicalLabelCache(
            root, sha256_file(manifest_path), expected_role_sha256=args[3]
        )


def test_fail_fast_success_keeps_full_accepted_coverage(monkeypatch, tmp_path):
    args = inputs(tmp_path)
    calls = _install_fake_encoder(
        monkeypatch,
        lambda source_index, _smiles, components: _ok_label(
            source_index, components
        ),
    )
    root = tmp_path / "cache"
    report = build_cache(*args, root, fail_fast=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert calls == [3, 9]
    assert report["accepted"] is True
    assert manifest["row_count"] == manifest["expected_row_count"] == 2
    assert manifest["export_row_count"] == 2
    assert manifest["export_coverage_complete"] is True
    assert manifest["failure_inventory_complete"] is True
    assert manifest["failures"] == []
    assert (root / "labels.npz").is_file()
