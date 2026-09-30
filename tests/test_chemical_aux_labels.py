from types import SimpleNamespace
import numpy as np
import pytest
from molgap.chemical_aux_labels import ChemicalLabelEncoder, compact_head_parameter_count


@pytest.fixture(scope="module")
def encoder():
    return ChemicalLabelEncoder()


@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "CC(=O)O", "[NH4+]", "C.C", "Cl", "C[C@H](O)C(=O)O"])
def test_matches_upstream_and_keeps_row_identity(encoder, smiles):
    from rdkit import Chem
    result = encoder.encode(73, smiles)
    assert result.status == "ok" and result.source_index == 73
    np.testing.assert_array_equal(result.descriptors, np.asarray(encoder.generator.process(smiles)[1:], dtype=np.float32))
    np.testing.assert_array_equal(result.fingerprint, np.asarray(Chem.RDKFingerprint(Chem.MolFromSmiles(smiles), minPath=1, maxPath=7, fpSize=512)))
    assert result.descriptors.shape == (200,) and np.isfinite(result.descriptors).all()


@pytest.mark.parametrize("smiles", ["", "   ", "not-a-molecule", None])
def test_invalid_row_is_not_silently_dropped(encoder, smiles):
    result = encoder.encode(81, smiles)
    assert result.source_index == 81 and result.status == "invalid_smiles"
    assert result.descriptors is None and result.fingerprint is None


@pytest.mark.parametrize("raw,status", [(None, "descriptor_failure"), ([False] + [0.0] * 200, "descriptor_failure"), ([True] + [float("nan")] * 200, "nonfinite_descriptor")])
def test_descriptor_failures_are_explicit(encoder, monkeypatch, raw, status):
    monkeypatch.setattr(encoder.generator, "process", lambda _: raw)
    assert encoder.encode(4, "CC").status == status


def test_static_contract(encoder):
    assert len(set(encoder.identity()["descriptor_names"])) == 200
    assert compact_head_parameter_count() == 32744
    with pytest.raises(ValueError):
        encoder.encode(-1, "CC")


def test_pcqm_topology_fp_only_accepts_hypervalent_si_row_51128(monkeypatch):
    from rdkit import Chem, DataStructs
    from molgap import pcqm_feature_screen

    smiles = "C[Si](C)(C)(C)(C)C"
    molecule = SimpleNamespace(GetNumAtoms=lambda: 7)
    topology_calls = []
    fingerprint_calls = []

    def topology_parse(value):
        topology_calls.append(value)
        return molecule, False

    def fingerprint(mol, **kwargs):
        fingerprint_calls.append((mol, kwargs))
        return object()

    def convert_to_array(_fingerprint, output):
        output[:] = 1

    monkeypatch.setattr(pcqm_feature_screen, "_molecule", topology_parse)
    monkeypatch.setattr(Chem, "RDKFingerprint", fingerprint)
    monkeypatch.setattr(DataStructs, "ConvertToNumpyArray", convert_to_array)

    encoder = ChemicalLabelEncoder(
        components=("fingerprints",), parse_policy="pcqm_topology"
    )
    result = encoder.encode(51128, smiles)

    assert result.status == "ok"
    assert result.source_index == 51128
    assert result.smiles_sanitized is False
    assert result.descriptors is None and result.descriptor_valid_mask is None
    assert result.fingerprint.shape == (512,)
    assert result.fingerprint.dtype == np.uint8
    assert result.fingerprint.all()
    assert topology_calls == [smiles]
    assert fingerprint_calls == [
        (molecule, {"minPath": 1, "maxPath": 7, "fpSize": 512})
    ]
    assert encoder.generator is None


def test_pcqm_topology_real_hypervalent_si_has_fingerprint_and_masked_qed():
    smiles = "O[Si]123O[Si]3(O1)(O2)O"
    encoder = ChemicalLabelEncoder(
        components=("descriptors", "fingerprints"),
        parse_policy="pcqm_topology",
        descriptor_missing_policy="mask_nonfinite",
    )
    result = encoder.encode(51128, smiles)

    assert result.status == "ok"
    assert result.source_index == 51128
    assert result.smiles_sanitized is False
    assert result.fingerprint.shape == (512,)
    assert result.fingerprint.dtype == np.uint8
    assert np.isin(result.fingerprint, [0, 1]).all()
    assert result.descriptors.shape == (200,)
    assert np.isfinite(result.descriptors).all()
    qed_index = encoder.descriptor_names.index("qed")
    assert not result.descriptor_valid_mask[qed_index]
    assert result.descriptors[qed_index] == 0


def test_fingerprint_only_does_not_construct_descriptor_generator(monkeypatch):
    from descriptastorus.descriptors import rdNormalizedDescriptors
    from rdkit import Chem, DataStructs

    calls = []
    molecule = SimpleNamespace(GetNumAtoms=lambda: 2)

    def forbidden_generator():
        calls.append("constructed")
        raise AssertionError("fingerprint-only mode must skip descriptors")

    monkeypatch.setattr(
        rdNormalizedDescriptors, "RDKit2DNormalized", forbidden_generator
    )
    monkeypatch.setattr(Chem, "MolFromSmiles", lambda _smiles: molecule)
    monkeypatch.setattr(Chem, "RDKFingerprint", lambda _mol, **_kwargs: object())
    monkeypatch.setattr(
        DataStructs, "ConvertToNumpyArray", lambda _fp, output: output.fill(1)
    )

    encoder = ChemicalLabelEncoder(components=("fingerprints",))
    result = encoder.encode(17, "CC")

    assert calls == []
    assert result.status == "ok" and result.source_index == 17
    assert result.descriptors is None
    assert result.fingerprint.shape == (512,)


def test_nonfinite_qed_can_be_rejected_or_masked(monkeypatch):
    from descriptastorus.descriptors import rdNormalizedDescriptors
    from rdkit import Chem

    molecule = SimpleNamespace(GetNumAtoms=lambda: 2)
    monkeypatch.setattr(Chem, "MolFromSmiles", lambda _smiles: molecule)

    class FakeGenerator:
        names = [f"descriptor_{i}" for i in range(200)]
        names[117] = "qed"

        def GetColumns(self):
            return [("smiles", None)] + [(name, None) for name in self.names]

        def process(self, _smiles):
            raw = [True] + [0.25] * 200
            raw[1 + self.names.index("qed")] = float("nan")
            return raw

    monkeypatch.setattr(
        rdNormalizedDescriptors, "RDKit2DNormalized", FakeGenerator
    )
    rejected_encoder = ChemicalLabelEncoder(
        components=("descriptors",),
        parse_policy="strict",
        descriptor_missing_policy="reject",
    )
    rejected = rejected_encoder.encode(11, "CC")
    assert rejected.status == "nonfinite_descriptor"
    assert rejected.descriptors is None

    masked_encoder = ChemicalLabelEncoder(
        components=("descriptors",),
        parse_policy="strict",
        descriptor_missing_policy="mask_nonfinite",
    )
    masked = masked_encoder.encode(11, "CC")
    qed_index = masked_encoder.descriptor_names.index("qed")
    assert masked.status == "ok"
    assert masked.descriptors.shape == (200,)
    assert np.isfinite(masked.descriptors).all()
    assert masked.descriptors[qed_index] == 0
    assert masked.descriptor_valid_mask.shape == (200,)
    assert masked.descriptor_valid_mask.dtype == np.bool_
    assert not masked.descriptor_valid_mask[qed_index]
    assert masked.descriptor_valid_mask.sum() == 199
