from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_forbids_promotion_and_sealed_roles():
    text = (ROOT / "experiments/pcqm_k1_explainability_audit/protocol.md").read_text(
        encoding="utf-8"
    )
    assert "cannot promote" in text
    assert "Do not construct a model" in text
    assert "official validation" in text
    assert "test roles" in text


def test_runtime_has_frozen_identities():
    from molgap.pcqm_k1_explainability import (
        DEVELOPMENT_ROWS,
        EXPECTED_GEOMETRY_SHA256,
        EXPECTED_MANIFEST_SHA256,
        REFERENCE,
    )

    assert DEVELOPMENT_ROWS == 50_000
    assert REFERENCE == "neural_atom_k1_v4"
    assert len(EXPECTED_MANIFEST_SHA256) == 64
    assert len(EXPECTED_GEOMETRY_SHA256) == 64


def test_kunshan_job_gates_pickle_dependency_and_preflight():
    text = (
        ROOT
        / "experiments/pcqm_k1_explainability_audit/run_error_audit_kunshan.slurm"
    ).read_text(encoding="utf-8")
    assert "src/molgap/pcqm_wedge.py" in text
    assert "preflight_error_audit.py" in text
    assert "attempts/${SLURM_JOB_ID}" in text
