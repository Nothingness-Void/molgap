"""Stage one verified shared source package as a private Kaggle input dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from molgap.experiment_package import verify_experiment_source_package


SOURCE_DATASET = "nvoid912/molgap-gptrans-centered-logits-100k-source-v1"
INITIAL_STATE_SHA256 = "9205fc0f0f97f1cc1cea84ab4bd24206274a00c7d84d26366497feee1710c20c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-package-identity", required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    manifest = verify_experiment_source_package(package, args.repo_root.resolve())
    if manifest["package_identity"] != args.expected_package_identity:
        raise RuntimeError("Independently pinned package identity changed")
    if sha256_file(args.initial_state) != INITIAL_STATE_SHA256:
        raise RuntimeError("Frozen initial-state artifact changed")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    with tarfile.open(package / "source.tar.gz", "r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute()
                    or any(part in {"", ".", ".."} for part in name.parts)):
                raise RuntimeError("Unsafe source archive member")
            destination = source.joinpath(*name.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError("Missing source archive member")
            with stream, destination.open("xb") as handle:
                shutil.copyfileobj(stream, handle)
    for name in ("SOURCE_COMMIT.txt", "SOURCE_ARCHIVE_SHA256.txt",
                 "SOURCE_FILES.json", "package_manifest.json", "experiment_spec.json"):
        shutil.copyfile(package / name, output / name)
    shutil.copyfile(package / "source.tar.gz", output / "source_payload.bin")
    shutil.copyfile(args.initial_state, output / "initial_state.pt")
    (output / "dataset-metadata.json").write_text(json.dumps({
        "title": "MolGap GPTrans centered logits 100K frozen source",
        "id": SOURCE_DATASET,
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"package_identity": manifest["package_identity"],
                      "source_commit": manifest["source_commit"],
                      "source_archive_sha256": manifest["archive_sha256"],
                      "dataset": SOURCE_DATASET}, sort_keys=True))


if __name__ == "__main__":
    main()
