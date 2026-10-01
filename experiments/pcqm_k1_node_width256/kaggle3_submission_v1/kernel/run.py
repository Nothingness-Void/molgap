"""Standard Kaggle source bootstrap. The staged launch config owns the arms."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import types

# Local preparation replaces this marker and binds the resulting entry bytes.
EXPECTED_LAUNCH_SHA256 = '8fa5ff7eff4f717c41960675875c3ee3646823448e21b53c76e1cbd938da83b3'


def main():
    mounted = Path("/kaggle/input")
    configs = list(mounted.rglob("experiment_launch.json"))
    if len(configs) != 1:
        raise RuntimeError("Expected one frozen experiment_launch.json source mount")
    launch = configs[0]
    if EXPECTED_LAUNCH_SHA256 is None or hashlib.sha256(launch.read_bytes()).hexdigest() != EXPECTED_LAUNCH_SHA256:
        raise RuntimeError("Launch configuration differs from the prepared entrypoint")
    config = json.loads(launch.read_text(encoding="utf-8"))
    archive = launch.parent / "source_payload.bin"
    with archive.open("rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed != config["expected_source_archive_sha256"]:
        raise RuntimeError("Frozen source archive hash mismatch")
    # Dependencies are platform bootstrap, never imported from another experiment.
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "numpy<2",
                    "torch-geometric==2.6.1", "ogb==1.3.6"], check=True)
    root = Path("/kaggle/temp/molgap-workflow")
    root.mkdir(parents=True, exist_ok=False)
    package, source = root / "package", root / "source"
    package.mkdir()
    source.mkdir()
    shutil.copyfile(archive, package / "source.tar.gz")
    for name in ("experiment_spec.json", "package_manifest.json", "SOURCE_COMMIT.txt",
                 "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"):
        shutil.copyfile(launch.parent / name, package / name)
    with tarfile.open(archive, "r:gz") as bundle:
        stream = bundle.extractfile("src/molgap/experiment_preflight.py")
        if stream is None:
            raise RuntimeError("Shared source extractor absent")
        with stream:
            code = stream.read()
    bootstrap = types.ModuleType("_frozen_source_bootstrap")
    bootstrap.__file__ = str(archive) + ":experiment_preflight.py"
    exec(compile(code, bootstrap.__file__, "exec"), bootstrap.__dict__)
    bootstrap._unpack(package, source)
    sys.path.insert(0, str(source / "src"))
    from molgap.experiment_package import verify_experiment_source_package
    from molgap.kaggle_pair_runtime import run_two_phase_pair
    manifest = verify_experiment_source_package(package)
    if manifest["package_identity"] != config["expected_package_identity"] or manifest["spec_identity"] != config["spec_identity"]:
        raise RuntimeError("Frozen launch/Spec/package binding mismatch")
    run_two_phase_pair(source_root=source, package_dir=package, input_root=mounted,
                       launch_path=launch, output=Path("/kaggle/working/experiment"))


if __name__ == "__main__":
    main()
