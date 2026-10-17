"""Static checks for the thin IMS CPU Modern2D export adapter."""

from pathlib import Path
import re
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
PBS = ROOT / "platforms/ims/pcqm_fixed_datasets/export_indexed_roles.pbs"


def test_git_archive_head_retains_lf_pbs_bytes_identical_to_git_blob(tmp_path):
    relative = PBS.relative_to(ROOT).as_posix()
    archive_path = tmp_path / "export_indexed_roles.tar"
    # Reproduce the Windows packaging setting that introduced scheduler-rejected CRLF.
    subprocess.run(
        ["git", "-c", "core.autocrlf=true", "archive", "--format=tar",
         "--output", str(archive_path), "HEAD", "--", relative],
        cwd=ROOT, check=True, capture_output=True,
    )
    with tarfile.open(archive_path, "r:") as archive:
        member = archive.getmember(relative)
        assert member.isfile()
        stream = archive.extractfile(member)
        assert stream is not None
        with stream:
            archived_bytes = stream.read()
        source_commit = archive.pax_headers["comment"]
    blob_bytes = subprocess.run(
        ["git", "cat-file", "blob", f"{source_commit}:{relative}"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    assert archived_bytes
    assert b"\r" not in archived_bytes
    assert archived_bytes == blob_bytes


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


def test_ims_qualification_metadata_pbs_is_bounded_and_has_lf_attributes():
    relative = "platforms/ims/pcqm_fixed_datasets/prepare_qualification_inputs.pbs"
    text = (ROOT / relative).read_text(encoding="utf-8")
    resources = "\n".join(line for line in text.splitlines() if line.startswith("#PBS -l"))
    assert "select=1:ncpus=16:mpiprocs=1:ompthreads=1" in resources
    assert "walltime=00:10:00" in resources
    assert not re.search(r"(?:ngpus|gpus)=(?!0(?:\s|:|$))", resources)
    assert "set -euo pipefail" in text
    assert "BOUNDARY=/lustre/home/users/sm2/chou" in text
    assert 'STAGE="${MOLGAP_METADATA_STAGE:?explicit immutable source/request stage required}"' in text

    helper = ('"$BOUNDARY/molgap/.venv/bin/python" '
              '"$STAGE/code/experiments/modern2d_model_integration/scripts/prepare_qualification_inputs.py" '
              '--request "$STAGE/request.json" --execute')
    assert text.count(helper) == 1
    assert text.count("--execute") == 1
    for prerequisite in (
        'test "$(realpath "$BOUNDARY")" = "$BOUNDARY"',
        'test "$(realpath "$STAGE")" = "$STAGE"',
        'test "$(realpath "$STAGE/request.json")" = "$STAGE/request.json"',
        'case "$STAGE" in "$BOUNDARY"/*)',
        "assert Path(runtime['python']) == boundary/'molgap/.venv/bin/python'",
        "assert all(Path(p).resolve().is_relative_to(boundary) for p in paths)",
    ):
        assert text.index(prerequisite) < text.index(helper)
    assert "platforms/ims/v4_submit/runtime_registry.json" in text
    assert "ims-a100-torch271-v1" in text
    assert "runtime['pythonpath']" in text
    assert "CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1" in text
    assert "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1" in text
    assert not any(fragment in text.lower() for fragment in (
        "family_training", "/training", "/train_", "--train", "encoder", "jsub",
        "official_validation", "independent_audit", "test-dev", "test-challenge",
    ))

    attributes = subprocess.run(
        ["git", "check-attr", "-z", "text", "eol", "--", relative],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout.split(b"\0")
    assert attributes == [relative.encode(), b"text", b"set", relative.encode(), b"eol", b"lf", b""]
