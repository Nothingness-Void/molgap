"""Static checks for the thin IMS CPU Modern2D export adapter."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PBS = ROOT / "platforms/ims/pcqm_fixed_datasets/export_indexed_roles.pbs"


def test_ims_export_pbs_is_bounded_cpu_only_and_uses_explicit_paths():
    text = PBS.read_text(encoding="utf-8")
    lower = text.lower()

    assert "#pbs -l select=1:ncpus=16:mpiprocs=1:ompthreads=1" in lower
    assert "#pbs -l walltime=02:00:00" in lower
    assert "ngpus=1" not in lower
    assert "cuda_visible_devices=-1" in lower

    assert "boundary=/lustre/home/users/sm2/chou" in lower
    assert 'stage="${molgap_export_stage:?explicit committed source stage required}"' in lower
    assert 'request="$stage/request.json"' in lower
    assert 'output="${molgap_export_output:?explicit separate selected export required}"' in lower
    assert 'test "$(realpath "$BOUNDARY")" = "$BOUNDARY"' in text
    assert 'test "$(realpath "$STAGE")" = "$STAGE"' in text
    assert 'test "$(realpath "$REQUEST")" = "$REQUEST"' in text
    assert 'test "$(realpath -m "$OUTPUT")" = "$OUTPUT"' in text
    assert 'case "$STAGE" in "$BOUNDARY"/*)' in text
    assert 'case "$OUTPUT" in "$BOUNDARY"/*)' in text


def test_ims_export_pbs_uses_registered_runtime_then_exports_and_verifies():
    text = PBS.read_text(encoding="utf-8")

    registry = "platforms/ims/v4_submit/runtime_registry.json"
    assert registry in text
    assert "ims-a100-torch271-v1" in text
    assert "runtime['python']" in text
    assert "runtime['pythonpath']" in text
    assert "$BOUNDARY/molgap/.venv/bin/python" in text

    export_call = '"$PYTHON" "$ENTRY" --request "$REQUEST" --boundary "$BOUNDARY" --execute'
    verify_call = ('"$PYTHON" "$ENTRY" --request "$REQUEST" --boundary "$BOUNDARY" --execute '
                   '--verify-acceptance-sha256 "$PIN"')
    assert text.index(export_call) < text.index('PIN="$(sha256sum "$OUTPUT/acceptance.json")"')
    assert text.index(verify_call) > text.index('PIN="$(sha256sum "$OUTPUT/acceptance.json")"')
    assert text.count('--execute') == 2
    assert 'printf \'CPU_EXPORT_VERIFIED acceptance_sha256=%s' in text

    forbidden_path_fragments = ("/training", "/train_", "official_validation", "test_roles", "protected")
    assert not any(fragment in text.lower() for fragment in forbidden_path_fragments)
