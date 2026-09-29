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
