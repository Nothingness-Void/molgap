"""Load the frozen shared bootstrap, then run one prospective T4x2 pair."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys
import tarfile
import types


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/k1_pair"))
    parser.add_argument("--source-root", type=Path, default=Path("/kaggle/temp/k1_source"))
    arguments = parser.parse_args()
    configs = list(arguments.input_root.rglob("pair_launch.json"))
    if len(configs) != 1:
        raise FileNotFoundError(f"Expected one frozen pair_launch.json, found {len(configs)}")
    config_path = configs[0]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    package = config_path.parent
    archive = package / config.get("source_archive", "source_payload.bin")
    with archive.open("rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed != config["expected_source_archive_sha256"]:
        raise RuntimeError("Frozen pair source archive changed")
    # The verified archive owns this existing stdlib-only extractor. Loading it
    # before family imports avoids depending on a host MolGap installation.
    with tarfile.open(archive, "r:gz") as bundle:
        stream = bundle.extractfile("src/molgap/experiment_preflight.py")
        if stream is None:
            raise FileNotFoundError("Shared source bootstrap missing from archive")
        with stream:
            bootstrap_source = stream.read()
    bootstrap = types.ModuleType("_frozen_molgap_bootstrap")
    bootstrap.__file__ = str(archive) + ":src/molgap/experiment_preflight.py"
    exec(compile(bootstrap_source, bootstrap.__file__, "exec"), bootstrap.__dict__)
    source_root = arguments.source_root.resolve()
    source_root.mkdir(parents=True, exist_ok=False)
    unpack_package = source_root.parent / "k1_package"
    unpack_package.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(archive, unpack_package / "source.tar.gz")
    for name in ("experiment_spec.json", "package_manifest.json", "SOURCE_COMMIT.txt",
                 "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"):
        shutil.copyfile(package / name, unpack_package / name)
    bootstrap._unpack(unpack_package, source_root)
    dependency_owner = runpy.run_path(str(source_root /
        "experiments/pcqm_gptrans_pair_norm_100k/run_candidates.py"))
    dependency_owner["install_dependencies"]()
    sys.path.insert(0, str(source_root / "src"))
    from molgap.experiment_package import verify_experiment_source_package
    from molgap.kaggle_pair_runtime import run_two_phase_pair
    # Package bindings and tracked source hashes are checked again by each
    # worker's RunContext, before loading molecular roles.
    verification = verify_experiment_source_package(unpack_package)
    if verification["package_identity"] != config["expected_package_identity"]:
        raise RuntimeError("Frozen pair source package identity changed")
    jobs = config["jobs"]
    path_flags = {"recipe", "initial-state"}
    for job in jobs:
        flags = job["arguments"]
        for key in path_flags:
            path = Path(flags[key])
            base = source_root if key == "recipe" else package
            flags[key] = str(path if path.is_absolute() else base / path)
        flags["package-dir"] = str(unpack_package)
        flags["spec"] = str(unpack_package / "experiment_spec.json")
        flags["input-root"] = str(arguments.input_root)
    run_two_phase_pair(source_root=source_root, jobs=jobs, output=arguments.output)


if __name__ == "__main__":
    main()
