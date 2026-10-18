"""Metadata-only indexed Modern2D feature-binding contracts.

These checks intentionally stop at the CPU preview boundary.  They do not
instantiate either model, derive tensors, or make a native qualification claim.
"""

from copy import deepcopy
from dataclasses import asdict
import gzip
import json

import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset

from molgap.common.wire import canonical_fingerprint
from molgap.data.indexed_inputs import IndexedInputPermit, IndexedRoleDataset
from molgap.esa_native import ESANativeConfig
from molgap.experiment_execution import MODERN2D_PREVIEW_INTERFACES
from molgap.family_training_data import DatasetSelection, TrainingData
from molgap.family_training_loaders import FamilyLoaders
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.gptrans_execution_tuning import ExecutionTuning
from molgap.modern2d_feature_binding import (
    BINDING_FIELDS,
    build_indexed_feature_binding,
    indexed_transform_identity,
    validate_feature_binding,
    validate_indexed_feature_binding,
    transform_configuration,
)
from molgap.ppgt_author import PPGTAuthorConfig
from molgap.pcqm_unified_roles import prepare_unified_roles
from molgap.research_memory.trace import file_digest, json_bytes
from test_pcqm_indexed_inputs import fixture as indexed_fixture
from test_pcqm_unified_roles import source_fixture as source_assets_fixture


CASES = (
    pytest.param(
        ("ppgt_small", "2"),
        PPGTAuthorConfig(
            hidden=8,
            heads=2,
            blocks=1,
            pair_dim=4,
            pe_mlp_hidden=8,
            stem_blocks=1,
            rrwp_steps=2,
            spe_bases=1,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            max_nodes=8,
            max_edges=16,
        ),
        id="ppgt-2",
    ),
    pytest.param(
        ("esa_small", "3"),
        ESANativeConfig(
            hidden=8,
            heads=2,
            layer_types=("M", "S", "P"),
            graph_dim=8,
            mlp_hidden=16,
            seeds=2,
            max_nodes=8,
            max_edges=16,
        ),
        id="esa-3",
    ),
)


@pytest.mark.parametrize("family,config", CASES)
def test_feature_configuration_ignores_width_but_not_topology(family, config):
    from dataclasses import replace
    assert transform_configuration(family, config) == transform_configuration(family, replace(config, hidden=16))
    assert transform_configuration(family, config) != transform_configuration(family, replace(config, max_nodes=9))


def test_feature_only_identity_still_pins_native_esa_tokenization(monkeypatch):
    from molgap import v4_runtime
    seen = []
    def source(path):
        seen.append(path.as_posix())
        return "a" * 64
    monkeypatch.setattr(v4_runtime, "normalized_source_sha256", source)
    indexed_transform_identity(("esa_small", "4"), feature_only=True)
    assert any(path.endswith("/esa_native.py") for path in seen)
    assert any(path.endswith("/esa_features.py") for path in seen)
    with pytest.raises(ValueError, match="Unreviewed"):
        transform_configuration(("unregistered", "1"), {})


def _dataset(tmp_path, role="train_100k"):
    root = tmp_path / role
    root.mkdir()
    indexed_root, permit, authority, history = indexed_fixture(root, role)
    return IndexedRoleDataset(
        indexed_root,
        permit,
        authority=authority,
        role_history=history,
    )


def _generated_128_dataset(tmp_path, role="train_500k"):
    """Build 133 tiny non-PCQM graphs so the synthetic train_500k view has 128 rows."""
    root = tmp_path / "generated_128"
    root.mkdir()
    fixed, rule = source_assets_fixture(root)
    total_rows = 133
    graphs = [
        Data(
            x=torch.ones((1, 9), dtype=torch.long),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 3), dtype=torch.long),
            random_walk_pe=torch.zeros((1, 16)),
            y=torch.tensor([float(index) / 10]),
            source_idx=torch.tensor([index], dtype=torch.long),
        )
        for index in range(total_rows)
    ]
    torch.save(InMemoryDataset.collate(graphs), fixed / "train.pt")
    with gzip.open(fixed / "rows.csv.gz", "wt") as stream:
        stream.write("idx,smiles,homolumogap\n")
        for index in range(total_rows):
            stream.write(f"{index},{'C' * (index + 1)},{float(index) / 10}\n")
        stream.write(f"{total_rows},DO_NOT_PARSE_OFFICIAL_VALIDATION,not-a-label\n")

    source_manifest_path = fixed / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_bytes())
    source_manifest["store_assets"]["source"][0]["sha256"] = file_digest(fixed / "rows.csv.gz")
    topology = source_manifest["store_assets"]["topology"][0]
    topology.update(
        sha256=file_digest(fixed / "train.pt"),
        rows=total_rows,
        source_idx_min=0,
        source_idx_max=total_rows - 1,
    )
    source_manifest_path.write_bytes(json_bytes(source_manifest))
    rule.update(
        official_train_rows=total_rows,
        train_sizes=[2, 128, 129],
        excluded_ranges=[],
        source_manifest_sha256=file_digest(source_manifest_path),
    )

    output = root / "indexed_128"
    acceptance = prepare_unified_roles(
        boundary=root,
        fixed_root=fixed,
        output_root=output,
        rule=rule,
    )
    manifest_path = output / "roles" / f"{role}.json"
    authority = {
        "scope": "CPU_SYNTHETIC_ONLY",
        "role": role,
        "manifest_sha256": file_digest(manifest_path),
        "operations": ["prediction_input", "labels_read"],
        "protected": False,
    }
    history = {
        "role": role,
        "manifest_sha256": file_digest(manifest_path),
        "usage": "training",
    }
    assert acceptance["roles"][role]["sha256"] == file_digest(manifest_path)
    permit = IndexedInputPermit(
        role,
        file_digest(output / "acceptance.json"),
        file_digest(manifest_path),
        acceptance["rule_identity"],
        canonical_fingerprint(authority),
        canonical_fingerprint(history),
    )
    return IndexedRoleDataset(output, permit, authority=authority, role_history=history)


def _training_data(dataset, recipe, source_indices):
    return TrainingData(
        dataset,
        None,
        recipe.dataset.manifest_sha256,
        recipe.train_rows,
        0,
        None,
        None,
        0.0,
        1.0,
        synthetic=True,
        train_source_indices=source_indices,
        development_source_indices=None,
        indexed=True,
    )


def _preview_recipe(family, configuration):
    selection = DatasetSelection("synthetic/modern2d_feature_probe.json", "a" * 64, False)
    return FamilyRecipe(
        family[0],
        "reference",
        42,
        "b" * 64,
        selection,
        128,
        0,
        1,
        0.001,
        0.01,
        ScheduleSpec("cosine_step", 1),
        "final-live",
        "torch-pass-randperm",
        checkpoint_every=1,
        target_mean_eV=3.0,
        target_std_eV=2.0,
        finite_check_every=1,
        model_config=configuration,
        family_version=family[1],
    )


@pytest.mark.parametrize("family,configuration", CASES)
def test_indexed_binding_freezes_all_input_and_transform_identities(
    tmp_path, family, configuration
):
    dataset = _dataset(tmp_path)
    binding = build_indexed_feature_binding(
        dataset,
        family=family,
        configuration=configuration,
    )
    manifest = dataset.manifest

    assert set(binding) == BINDING_FIELDS
    assert binding == {
        "base_dataset_manifest_sha": dataset.permit.manifest_sha256,
        "source_row_sha": manifest["source_index_sha256"],
        "source_topology_sha": canonical_fingerprint(
            {"chunks": [chunk["topology_sha256"] for chunk in manifest["graph_chunks"]]}
        ),
        "feature_schema_version": MODERN2D_PREVIEW_INTERFACES[family][1],
        "transform_code_sha": indexed_transform_identity(family),
        "transform_config_sha": canonical_fingerprint(asdict(configuration)),
        "derived_feature_sha": None,
        "role_history_sha": dataset.permit.role_history_sha256,
        "role": dataset.permit.role,
        "scope": dataset.permit.scope,
    }
    assert binding["derived_feature_sha"] is None
    assert validate_indexed_feature_binding(binding, deepcopy(binding)) == canonical_fingerprint(
        binding
    )


@pytest.mark.parametrize("family,configuration", CASES)
def test_metadata_binding_never_reads_nan_targets_or_rwse(
    tmp_path, monkeypatch, family, configuration
):
    dataset = _dataset(tmp_path)
    assert dataset._loaded is not None
    dataset._loaded._data.y.fill_(float("nan"))
    dataset._loaded._data.random_walk_pe.fill_(float("nan"))

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata binding read graph payload or derived features")

    monkeypatch.setattr(IndexedRoleDataset, "_load", forbidden)
    monkeypatch.setattr(IndexedRoleDataset, "__getitem__", forbidden)
    monkeypatch.setattr(IndexedRoleDataset, "selected_targets", forbidden)

    binding = build_indexed_feature_binding(
        dataset,
        family=family,
        configuration=configuration,
    )
    assert binding["derived_feature_sha"] is None


@pytest.mark.parametrize("family,configuration", CASES)
def test_family_loaders_exposes_bindings_for_actual_indexed_objects(
    tmp_path, family, configuration
):
    dataset = _generated_128_dataset(tmp_path)
    recipe = _preview_recipe(family, configuration)

    assert len(dataset) == recipe.train_rows == 128
    assert dataset.source_indices == tuple(range(recipe.train_rows))
    data = _training_data(dataset, recipe, dataset.source_indices)
    loaders = FamilyLoaders(recipe, data, ExecutionTuning(training_workers=0))
    try:
        assert set(loaders.feature_bindings) == {"train"}
        assert loaders.feature_bindings["train"] == build_indexed_feature_binding(
            dataset,
            family=family,
            configuration=configuration,
        )
        # This is a bounded CPU DataLoader check over generated packed input,
        # not a Session/model/native-adapter qualification.
        batch = next(iter(loaders.training))
        expected_order = torch.randperm(
            recipe.train_rows,
            generator=torch.Generator().manual_seed(recipe.seed),
        ).tolist()
        assert batch.source_idx.view(-1).tolist() == [
            dataset.source_indices[index] for index in expected_order
        ]
    finally:
        loaders.close()


@pytest.mark.parametrize("family,configuration", CASES)
def test_family_loaders_rejects_two_decoded_rows_declared_as_128(
    tmp_path, family, configuration
):
    dataset = _dataset(tmp_path)
    recipe = _preview_recipe(family, configuration)
    assert len(dataset) == 2
    with pytest.raises(ValueError, match="decoded row count"):
        FamilyLoaders(
            recipe,
            _training_data(dataset, recipe, tuple(range(recipe.train_rows))),
            ExecutionTuning(training_workers=0),
        )


@pytest.mark.parametrize("family,configuration", CASES)
def test_family_loaders_rejects_indexed_metadata_source_id_mismatch(
    tmp_path, family, configuration
):
    dataset = _generated_128_dataset(tmp_path)
    recipe = _preview_recipe(family, configuration)
    assert len(dataset) == recipe.train_rows == 128
    mismatched = dataset.source_indices[:-1] + (dataset.source_indices[-1] + 1,)
    assert len(mismatched) == len(dataset.source_indices)
    with pytest.raises(ValueError, match="verified dataset"):
        FamilyLoaders(
            recipe,
            _training_data(dataset, recipe, mismatched),
            ExecutionTuning(training_workers=0),
        )


@pytest.mark.parametrize("family,configuration", CASES)
def test_unknown_legacy_family_is_not_an_indexed_preview(
    tmp_path, family, configuration
):
    dataset = _dataset(tmp_path)
    for legacy in (("ppgt_small", "1"), ("esa_small", "1"), ("esa_small", "2")):
        with pytest.raises(ValueError, match="Unreviewed|explicit new model version"):
            build_indexed_feature_binding(
                dataset,
                family=legacy,
                configuration=configuration,
            )


@pytest.mark.parametrize("family,configuration", CASES)
def test_protected_role_materialized_cache_and_repinned_schema_are_rejected(
    tmp_path, family, configuration
):
    dataset = _dataset(tmp_path)
    binding = build_indexed_feature_binding(
        dataset,
        family=family,
        configuration=configuration,
    )

    protected = {**binding, "role": "independent_audit"}
    with pytest.raises(ValueError):
        validate_indexed_feature_binding(protected, protected)

    materialized = {**binding, "derived_feature_sha": "c" * 64}
    with pytest.raises(ValueError, match="materialized"):
        validate_indexed_feature_binding(materialized, materialized)

    repinned = {
        **binding,
        "feature_schema_version": "ogb-atom9-bond3-rwse16-v1",
    }
    with pytest.raises(ValueError, match="schema|Unreviewed"):
        validate_indexed_feature_binding(repinned, repinned)


@pytest.mark.parametrize("family,configuration", CASES)
def test_every_indexed_binding_field_is_frozen(tmp_path, family, configuration):
    dataset = _dataset(tmp_path)
    binding = build_indexed_feature_binding(
        dataset,
        family=family,
        configuration=configuration,
    )
    for field in sorted(BINDING_FIELDS):
        changed = deepcopy(binding)
        if field.endswith("_sha"):
            changed[field] = "d" * 64
        elif field == "feature_schema_version":
            changed[field] = "tampered-modern2d-schema"
        elif field == "role":
            changed[field] = "common_dev"
        else:
            changed[field] = "ACCEPTED_INDEXED_INPUT_V1"
        with pytest.raises(ValueError):
            validate_indexed_feature_binding(changed, binding)


@pytest.mark.parametrize("family,configuration", CASES)
@pytest.mark.parametrize("location", ["disk", "memory"])
def test_tampered_indexed_manifest_is_denied(
    tmp_path, family, configuration, location
):
    dataset = _dataset(tmp_path)
    manifest_path = dataset.root / "roles" / f"{dataset.permit.role}.json"
    changed = deepcopy(dataset.manifest)
    changed["source_index_sha256"] = "e" * 64
    if location == "disk":
        manifest_path.write_bytes(json_bytes(changed))
    else:
        dataset.manifest = changed

    with pytest.raises(ValueError, match="manifest|input"):
        build_indexed_feature_binding(
            dataset,
            family=family,
            configuration=configuration,
        )


@pytest.mark.parametrize("family,configuration", CASES)
def test_old_synthetic_feature_validator_keeps_its_cpu_role_boundary(
    tmp_path, family, configuration
):
    dataset = _dataset(tmp_path)
    binding = build_indexed_feature_binding(
        dataset,
        family=family,
        configuration=configuration,
    )
    with pytest.raises(ValueError, match="Real/protected"):
        validate_feature_binding(binding, binding)


@pytest.mark.parametrize("family,configuration", CASES)
def test_source_closure_hash_changes_preview_recipe_identity_and_resume_policy(
    tmp_path, monkeypatch, family, configuration
):
    del tmp_path
    recipe = _preview_recipe(family, configuration)
    serialized = json.loads(json.dumps(recipe.to_dict()))
    original_transform = recipe.execution_policy["feature_transform_source_sha256"]
    original_identity = recipe.identity

    import molgap.v4_runtime as v4_runtime

    monkeypatch.setattr(
        v4_runtime,
        "normalized_source_sha256",
        lambda path: "f" * 64,
    )

    assert indexed_transform_identity(family) != original_transform
    assert recipe.execution_policy["feature_transform_source_sha256"] != original_transform
    assert recipe.identity != original_identity
    with pytest.raises(ValueError, match="Execution policy"):
        FamilyRecipe.from_dict(serialized)
