"""Focused synthetic indexed-input and development-evaluator contracts."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import pickle
from types import SimpleNamespace

import pytest
import torch

from molgap.common.wire import canonical_fingerprint
from molgap.data.indexed_inputs import (
    AcceptedIndexedInputPermit,
    IndexedInputPermit,
    IndexedRoleDataset,
)
from molgap.family_training_data import (
    DatasetSelection,
    load_indexed_training_data,
)
from molgap.research_memory.trace import file_digest, json_bytes
from molgap.training.core import evaluate_development
from test_pcqm_indexed_inputs import fixture as indexed_fixture


def _role_records(root):
    receipt = json.loads((root / "acceptance.json").read_bytes())
    records = {}
    for role in ("train_100k", "train_500k", "common_dev"):
        manifest = root / receipt["roles"][role]["path"]
        authority = {
            "scope": "CPU_SYNTHETIC_ONLY",
            "role": role,
            "manifest_sha256": file_digest(manifest),
            "operations": ["prediction_input", "labels_read"],
            "protected": False,
        }
        history = {
            "role": role,
            "manifest_sha256": file_digest(manifest),
            "usage": "training" if role.startswith("train_") else "selection-used",
        }
        permit = IndexedInputPermit(
            role,
            file_digest(root / "acceptance.json"),
            file_digest(manifest),
            receipt["rule_identity"],
            canonical_fingerprint(authority),
            canonical_fingerprint(history),
        )
        records[role] = (permit, authority, history)
    return records


def _binding(tmp_path, train_role="train_100k", development=True):
    fixture_root = tmp_path / "synthetic"
    fixture_root.mkdir()
    root, _, _, _ = indexed_fixture(fixture_root, train_role)
    records = _role_records(root)
    path = fixture_root / "indexed-training-input.json"
    payload = {
        "format": "molgap-indexed-training-input-v1",
        "data_root": root.relative_to(path.parent).as_posix(),
        "scope": "CPU_SYNTHETIC_ONLY",
        "train": {
            "permit": asdict(records[train_role][0]),
            "authority": records[train_role][1],
            "role_history": records[train_role][2],
        },
        # Keep the original binding intact for a training-only selection; the
        # selection flag controls whether this role is opened.
        "development": {
            "permit": asdict(records["common_dev"][0]),
            "authority": records["common_dev"][1],
            "role_history": records["common_dev"][2],
        },
    }
    path.write_bytes(json_bytes(payload))
    selection = DatasetSelection(path.relative_to(tmp_path).as_posix(), file_digest(path), development)
    return root, path, selection, payload, records


def _rewrite_binding(path, selection, payload):
    path.write_bytes(json_bytes(payload))
    return replace(selection, manifest_sha256=file_digest(path))


@pytest.mark.parametrize("source_attachment", [False, True])
def test_indexed_training_data_returns_explicit_source_indices_and_train_only_normalization(tmp_path, source_attachment):
    root, binding_path, selection, _, _ = _binding(tmp_path)
    if source_attachment:
        attachment = tmp_path / "source/acceptance" / binding_path.name
        attachment.parent.mkdir(parents=True)
        attachment.write_bytes(binding_path.read_bytes())
        selection = replace(selection, manifest=binding_path.name)
    data = load_indexed_training_data(tmp_path, selection)

    assert data.indexed is True
    assert data.synthetic is True
    assert data.train_source_start is None
    assert data.development_source_start is None
    assert data.train_source_indices == tuple(data.train.source_indices)
    assert data.development_source_indices == tuple(data.development.source_indices)
    assert len(data.train_source_indices) == data.train_rows
    assert len(data.development_source_indices) == data.development_rows
    assert data.development_source_indices != tuple(
        range(data.development_source_indices[0], data.development_source_indices[0] + data.development_rows)
    )

    train_targets = torch.cat(list(data.train.selected_targets()))
    assert data.target_mean_eV == pytest.approx(float(train_targets.mean()))
    assert data.target_std_eV == pytest.approx(float(train_targets.std().clamp_min(1e-6)))
    # The fixture's development labels are deliberately different; they must
    # not affect the train-only normalization statistics.
    all_targets = torch.cat([train_targets, torch.cat(list(data.development.selected_targets()))])
    assert data.target_mean_eV != pytest.approx(float(all_targets.mean()))
    assert root.exists()
    if source_attachment:
        from molgap.family_training_data import resolve_pinned_manifest
        import shutil
        shutil.copytree(root, attachment.parent / root.name)
        with pytest.raises(ValueError, match="uniquely mounted pinned manifest"):
            resolve_pinned_manifest(tmp_path, selection)


def test_indexed_loader_does_not_read_audit_or_identity_files(tmp_path):
    root, _, selection, _, _ = _binding(tmp_path)
    for name in (
        "private_identity.sqlite",
        "preparation_role_events.json",
        "rule.json",
        "progress.json",
    ):
        path = root / name
        assert path.exists()
        path.unlink()

    data = load_indexed_training_data(tmp_path, selection)
    assert data.indexed and data.train_rows > 0 and data.development_rows > 0


def test_training_only_indexed_selection_keeps_binding_dev_but_does_not_open_it(tmp_path):
    root, _, selection, _, _ = _binding(tmp_path, development=False)
    dev_manifest = json.loads((root / "roles/common_dev.json").read_bytes())
    dev_payload = root / dev_manifest["graph_chunks"][0]["file"]
    dev_payload.unlink()

    data = load_indexed_training_data(tmp_path, selection)
    assert data.development is None
    assert data.development_rows == 0
    assert data.development_source_indices is None


@pytest.mark.parametrize(
    "kind",
    [
        "acceptance_sha256",
        "manifest_sha256",
        "version_identity",
        "authority_sha256",
        "role_history_sha256",
        "authority",
        "history",
    ],
)
def test_indexed_binding_rejects_source_pin_authority_and_history_tampering(
    tmp_path, monkeypatch, kind
):
    root, path, selection, payload, _ = _binding(tmp_path)
    changed = deepcopy(payload)
    record = changed["train"]
    if kind in {"acceptance_sha256", "manifest_sha256", "version_identity",
                "authority_sha256", "role_history_sha256"}:
        record["permit"][kind] = "e" * 64
    elif kind == "authority":
        record["authority"]["protected"] = True
    else:
        record["role_history"]["usage"] = "unbound"
    selection = _rewrite_binding(path, selection, changed)

    import molgap.data.packed_graph as packed_graph

    def forbidden(*args, **kwargs):
        raise AssertionError("tampered indexed input reached payload open")

    monkeypatch.setattr(packed_graph, "PackedGraphDataset", forbidden)
    with pytest.raises(ValueError):
        load_indexed_training_data(tmp_path, selection)
    assert root.exists()


def test_indexed_binding_denies_version_mismatch(tmp_path):
    _, path, selection, payload, _ = _binding(tmp_path)
    changed = deepcopy(payload)
    changed["development"]["permit"]["version_identity"] = "f" * 64
    selection = _rewrite_binding(path, selection, changed)
    with pytest.raises(ValueError):
        load_indexed_training_data(tmp_path, selection)


def test_indexed_binding_denies_train_development_source_overlap(tmp_path, monkeypatch):
    _, path, selection, payload, _ = _binding(tmp_path)
    _rewrite_binding(path, selection, payload)

    import molgap.data.indexed_inputs as indexed_inputs

    class OverlappingDataset:
        def __init__(self, root, permit, *, authority, role_history):
            self.permit = permit
            self.source_indices = (7, 19)

        def __len__(self):
            return len(self.source_indices)

        def selected_targets(self):
            yield torch.tensor([1.0, 2.0])

    monkeypatch.setattr(indexed_inputs, "IndexedRoleDataset", OverlappingDataset)
    with pytest.raises(ValueError, match="overlap"):
        load_indexed_training_data(tmp_path, selection)


def test_indexed_dataset_deepcopy_and_worker_serialization_omit_loaded_chunk(tmp_path):
    root, _, _, _, records = _binding(tmp_path, development=False)
    permit, authority, history = records["train_100k"]
    dataset = IndexedRoleDataset(root, permit, authority=authority, role_history=history)
    source = int(dataset[0].source_idx.item())
    assert dataset._loaded is not None

    state = dataset.__getstate__()
    assert state["_loaded"] is None
    assert state["_loaded_index"] is None
    assert state["_loaded_file_identity"] is None

    clones = [deepcopy(dataset), pickle.loads(pickle.dumps(dataset))]
    for clone in clones:
        assert clone._loaded is None
        assert clone._loaded_index is None
        assert int(clone[0].source_idx.item()) == source


@pytest.mark.parametrize(
    "role,expected_rows",
    [("train_100k", 100_000), ("train_500k", 500_000), ("common_dev", 50_000)],
)
def test_accepted_real_permit_rejects_wrong_count_before_payload_open(
    tmp_path, monkeypatch, role, expected_rows
):
    """The generated fixture is not real acceptance; it exercises the count gate."""
    root, _, _, _, records = _binding(tmp_path, development=False)
    receipt_path = root / "acceptance.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt.update(scope="OFFICIAL_TRAIN_CPU_PREPARATION_ONLY", status="DATA_ROLE_ACCEPTED")
    receipt_path.write_bytes(json_bytes(receipt))
    manifest_path = root / receipt["roles"][role]["path"]
    manifest_sha = file_digest(manifest_path)
    max_chunk_bytes = 16 * 1024 * 1024
    authority = {
        "scope": "ACCEPTED_INDEXED_INPUT_V1",
        "role": role,
        "acceptance_sha256": file_digest(receipt_path),
        "manifest_sha256": manifest_sha,
        "version_identity": receipt["rule_identity"],
        "operations": ["prediction_input", "labels_read"],
        "protected": False,
        "max_chunk_bytes": max_chunk_bytes,
    }
    history = {
        "role": role,
        "manifest_sha256": manifest_sha,
        "usage": "training" if role.startswith("train_") else "selection-used",
    }
    permit = AcceptedIndexedInputPermit(
        role,
        file_digest(receipt_path),
        manifest_sha,
        receipt["rule_identity"],
        canonical_fingerprint(authority),
        canonical_fingerprint(history),
        max_chunk_bytes,
    )
    assert json.loads(manifest_path.read_bytes())["rows"] < expected_rows

    import molgap.data.packed_graph as packed_graph

    opened = []

    def forbidden(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("wrong-count accepted permit opened a payload")

    monkeypatch.setattr(packed_graph, "PackedGraphDataset", forbidden)
    with pytest.raises(ValueError, match="frozen scale"):
        IndexedRoleDataset(root, permit, authority=authority, role_history=history)
    assert opened == []


def _evaluation_batches(source_indices, *, observed_indices=None):
    observed_indices = source_indices if observed_indices is None else observed_indices
    values = torch.arange(len(source_indices), dtype=torch.float32) + 1
    return [SimpleNamespace(
        y=values,
        source_idx=torch.as_tensor(observed_indices),
    )]


def test_evaluator_accepts_finite_explicit_noncontiguous_source_order():
    batches = _evaluation_batches((3, 17, 41))
    calls = []

    def forward(model, batch):
        calls.append(True)
        return batch.y.clone()

    result = evaluate_development(
        object(), batches, forward=forward, rows=3, source_start=None,
        source_indices=(3, 17, 41), mean=0., std=1.,
    )
    assert len(calls) == 1
    assert result["source_idx"].tolist() == [3, 17, 41]
    assert result["prediction_eV"].tolist() == [1., 2., 3.]
    assert result["mae_eV"] == pytest.approx(0.)
    assert torch.isfinite(result["prediction_eV"]).all()


def test_evaluator_retains_old_contiguous_source_start_compatibility():
    result = evaluate_development(
        object(), _evaluation_batches((12, 13)),
        forward=lambda model, batch: batch.y,
        rows=2, source_start=12, mean=0., std=1.,
    )
    assert result["source_idx"].tolist() == [12, 13]
    assert result["mae_eV"] == pytest.approx(0.)


@pytest.mark.parametrize(
    "bad_source_indices",
    [
        (3, 3),
        (True, 7),
        (3.0, 7),
        torch.tensor([3, 7], dtype=torch.int32),
        torch.tensor([3.0, 7.0]),
    ],
)
def test_evaluator_rejects_inferred_or_nonexact_explicit_source_ids_before_forward(
    bad_source_indices,
):
    calls = []

    def forbidden(model, batch):
        calls.append(True)
        raise AssertionError("invalid source identity reached forward")

    with pytest.raises(ValueError):
        evaluate_development(
            object(), _evaluation_batches((3, 7)), forward=forbidden,
            rows=2, source_start=None, source_indices=bad_source_indices,
            mean=0., std=1.,
        )
    assert calls == []


def test_evaluator_does_not_allow_source_start_to_infer_noncontiguous_identity():
    calls = []

    def forward(model, batch):
        calls.append(True)
        return batch.y

    with pytest.raises(ValueError, match="count/order"):
        evaluate_development(
            object(), _evaluation_batches((3, 17)), forward=forward,
            rows=2, source_start=3, mean=0., std=1.,
        )
    assert calls == [True]


def test_evaluator_rejects_tampered_observed_source_order_after_forward():
    calls = []

    def forward(model, batch):
        calls.append(True)
        return batch.y

    with pytest.raises(ValueError, match="count/order"):
        evaluate_development(
            object(), _evaluation_batches((3, 17, 41), observed_indices=(3, 41, 17)),
            forward=forward, rows=3, source_start=None,
            source_indices=(3, 17, 41), mean=0., std=1.,
        )
    assert calls == [True]


@pytest.mark.parametrize(
    "observed,expected",
    [(torch.tensor([3., 7.]), (3, 7)), (torch.tensor([False, True]), (0, 1))],
)
def test_evaluator_rejects_noninteger_observed_source_ids(observed, expected):
    batch = SimpleNamespace(y=torch.tensor([1., 2.]), source_idx=observed)
    with pytest.raises(ValueError):
        evaluate_development(
            object(), [batch], forward=lambda model, value: value.y,
            rows=2, source_start=None, source_indices=expected,
            mean=0., std=1.,
        )
