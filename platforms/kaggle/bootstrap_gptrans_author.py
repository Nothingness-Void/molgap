"""Reusable hash-pinned bootstrap for one configured native GPTrans screen."""
from pathlib import Path
import hashlib
import json
import sys
import tarfile
import types

SOURCE_SHA256 = "__PIN_SOURCE_ARCHIVE_SHA256__"


def main():
    matches = [p for p in Path("/kaggle/input").rglob("source_payload.bin")
               if hashlib.sha256(p.read_bytes()).hexdigest() == SOURCE_SHA256]
    if len(matches) != 1:
        raise RuntimeError("Missing or ambiguous frozen source archive")
    archive = matches[0]
    with tarfile.open(archive, "r:gz") as bundle:
        configs = [p for p in bundle.getnames() if p.endswith("/gpu/screen_config.json")]
        if len(configs) != 1:
            raise RuntimeError("Expected exactly one native screen configuration")
        config_ref = configs[0]
        config = json.loads(bundle.extractfile(config_ref).read())
        output_name = config["output_subdirectory"]
        if not output_name or Path(output_name).name != output_name:
            raise RuntimeError("Unsafe output directory")
        extractor = bundle.extractfile("src/molgap/experiment_preflight.py").read()
    output = Path("/kaggle/working") / output_name
    source = output / "verified_source"
    package = output / "source_package"
    # Reuse the shared confinement/hash-aware extractor from verified archive bytes.
    bootstrap = types.ModuleType("_frozen_source_bootstrap")
    bootstrap.__file__ = str(archive) + ":experiment_preflight.py"
    exec(compile(extractor, bootstrap.__file__, "exec"), bootstrap.__dict__)
    package.mkdir(parents=True, exist_ok=False)
    import shutil
    shutil.copyfile(archive, package / "source.tar.gz")
    for name in ("experiment_spec.json", "package_manifest.json", "SOURCE_COMMIT.txt",
                 "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json"):
        shutil.copyfile(archive.parent / name, package / name)
    bootstrap._unpack(package, source)
    sys.path.insert(0, str(source / "src"))
    from molgap.gptrans_author_screen import run_author_screen
    # The screen restores/verifies its own retained package through the native owner.
    run_author_screen(archive.parent, source, output, SOURCE_SHA256, config_ref=config_ref,
                      package_dir=package)


if __name__ == "__main__":
    main()
