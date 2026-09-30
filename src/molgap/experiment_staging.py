"""Local package-to-upload assembly; no planning, submission or compute authority."""
from __future__ import annotations

from dataclasses import dataclass
import re
import shutil
from pathlib import Path, PureWindowsPath
import tarfile

from .experiment_package import (
    SIDECARS, _no_links, _source, _spec, build_experiment_source_package,
)
from .experiment_preflight import check_release_inputs
from .research_memory.trace import atomic_write, file_digest, json_bytes

SOURCE_PIN = b"__PIN_SOURCE_ARCHIVE_SHA256__"
_MOUNT_SIDECARS = (
    "SOURCE_COMMIT.txt", "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json",
    "experiment_spec.json",
)


@dataclass(frozen=True)
class UploadArtifact:
    """Explicit trusted input, pinned separately from source and tensor identities."""

    path: Path
    sha256: str


def _artifact_name(name: str) -> str:
    if (type(name) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            or name.endswith(".") or PureWindowsPath(name).is_reserved() or name.casefold() in {
                *(n.casefold() for n in SIDECARS), "source_payload.bin", "dataset-metadata.json",
            }
            or set(re.split(r"[._-]+", name.casefold())) & {
                "key", "keys", "secret", "secrets", "credential", "credentials",
                "password", "passwords", "token", "tokens",
            }):
        raise ValueError(f"Unsafe or reserved upload artifact name: {name}")
    return name


def _trusted_file(path: Path) -> Path:
    path = Path(path).absolute()
    _no_links(path)
    return _source(path.parent.resolve(), path.name)


def stage_release_inputs(
    spec, repo_root: Path, relative_paths, output: Path, *,
    artifacts: dict[str, UploadArtifact], recipe_files: dict[str, str],
    initial_states: dict[str, str], required_modules,
    entry_template: Path, kernel_metadata: Path,
    dataset_metadata: dict | None = None, pickle_inputs=(),
) -> dict:
    """Assemble once and run the existing release check against actual upload bytes.

    ``initial_states`` maps arm IDs to artifact names, not independent paths.
    Metadata belongs to the platform caller; this function never chooses an
    account, accelerator, scientific protocol or RML outcome. Failed/partial
    output is retained and cannot be overwritten on retry.
    """
    spec = _spec(spec)
    repo_root = Path(repo_root).resolve()
    relative_paths = list(relative_paths)
    output = Path(output).absolute()
    _no_links(output)
    if output.exists():
        raise FileExistsError(output)
    names = [_artifact_name(name) for name in artifacts]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Case-colliding upload artifacts")
    arm_ids = {arm["arm_id"] for arm in spec.to_dict()["arms"]}
    if set(recipe_files) != arm_ids or set(initial_states) - arm_ids:
        raise ValueError("Recipe/initialization arm bindings differ from Spec")
    if set(initial_states.values()) - set(names):
        raise ValueError("Initialization must name a staged artifact")
    template_path, metadata_path = _trusted_file(entry_template), _trusted_file(kernel_metadata)
    for path in (template_path, metadata_path):
        if (not path.is_relative_to(repo_root)
                or path.relative_to(repo_root).as_posix() not in relative_paths):
            raise ValueError("Entry template and kernel metadata must be explicit packaged source")
    template = template_path.read_bytes().replace(b"\r\n", b"\n")
    if template.count(SOURCE_PIN) != 1:
        raise ValueError("Entry template must have exactly one source archive pin")
    trusted = {}
    for name, artifact in artifacts.items():
        if (type(artifact) is not UploadArtifact or type(artifact.sha256) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", artifact.sha256)):
            raise ValueError("Upload inputs require an explicit SHA256")
        path = _trusted_file(artifact.path)
        if path.is_relative_to(output.resolve()) or file_digest(path) != artifact.sha256:
            raise ValueError(f"Input file SHA/boundary mismatch: {name}")
        trusted[name] = path
    output.mkdir(parents=True)
    source, inputs, kernel = (output / name for name in ("source", "inputs", "kernel"))
    try:
        package = build_experiment_source_package(spec, repo_root, relative_paths, source)
        # Render from immutable archive bytes, not a file that may change after freeze.
        with tarfile.open(source / "source.tar.gz", "r:gz") as archive:
            template_stream = archive.extractfile(template_path.relative_to(repo_root).as_posix())
            metadata_stream = archive.extractfile(metadata_path.relative_to(repo_root).as_posix())
            if template_stream is None or metadata_stream is None:
                raise ValueError("Missing packaged entry/template metadata")
            template, metadata = template_stream.read(), metadata_stream.read()
        if template.count(SOURCE_PIN) != 1:
            raise ValueError("Frozen entry template must contain exactly one source pin")
        inputs.mkdir()
        kernel.mkdir()
        for name in _MOUNT_SIDECARS:
            shutil.copyfile(source / name, inputs / name)
        shutil.copyfile(source / "source.tar.gz", inputs / "source_payload.bin")
        for name, path in trusted.items():
            shutil.copyfile(path, inputs / name)
            if file_digest(inputs / name) != artifacts[name].sha256:
                raise ValueError(f"Input changed during staging: {name}")
        if dataset_metadata is not None:
            atomic_write(inputs / "dataset-metadata.json", json_bytes(dataset_metadata))
        atomic_write(kernel / "run.py", template.replace(SOURCE_PIN, package["archive_sha256"].encode()))
        atomic_write(kernel / "kernel-metadata.json", metadata)
        report = check_release_inputs(
            spec, source, expected_package_identity=package["package_identity"],
            recipe_files=recipe_files,
            initial_states={arm: inputs / name for arm, name in initial_states.items()},
            required_modules=required_modules, pickle_inputs=pickle_inputs,
            entry_script=kernel / "run.py", input_root=inputs,
        )
        atomic_write(output / "release.json", json_bytes(report))
        result = {
            "format": "molgap-local-release-staging-v1", "package": package,
            "release_status": report["status"], "errors": report["errors"],
            "upload_artifacts": {name: file_digest(inputs / name) for name in names},
            "entry_script_sha256": file_digest(kernel / "run.py"),
            "kernel_metadata_sha256": file_digest(kernel / "kernel-metadata.json"),
            "dataset_metadata_sha256": file_digest(inputs / "dataset-metadata.json") if dataset_metadata is not None else None,
            "compute_released": False, "submitted": False,
        }
        atomic_write(output / "staging.json", json_bytes(result))
        return result
    except Exception as exc:
        atomic_write(output / "staging_failure.json", json_bytes({
            "status": "STAGING_FAILED", "error": str(exc),
            "compute_released": False, "submitted": False,
        }))
        raise
