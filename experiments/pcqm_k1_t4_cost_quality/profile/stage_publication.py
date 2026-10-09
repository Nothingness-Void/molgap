"""Stage exact-byte diagnostic payload for the existing Kaggle submitter."""
import argparse
import gzip
import io
import json
from pathlib import Path
import subprocess
import tarfile

from molgap.experiment_launch import publish_immutable_bytes
from molgap.k1_execution_profile import verify_t4_payload
from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = "molgap-k1-native-t4-profile-s42-v1"
DATASET = "nvoid912/molgap-k1-native-t4-profile-source-v1"


def stage(payload: Path, expected_digest: str, output: Path):
    manifest = verify_t4_payload(payload, expected_digest)
    if output.exists():
        raise FileExistsError("Publication staging must be fresh")
    output.mkdir(parents=True)
    dataset, kernel = output / "source_dataset", output / "kernel"
    dataset.mkdir()
    kernel.mkdir()
    # Preserve the already accepted sample bytes; do not deserialize or rebuild.
    names = sorted([*manifest["files"], "payload_manifest.json"])
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name in names:
                data = (payload / name).read_bytes()
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                archive.addfile(info, io.BytesIO(data))
    publish_immutable_bytes(dataset / "source_payload.bin", buffer.getvalue())
    publish_immutable_bytes(dataset / "payload_manifest.json", (payload / "payload_manifest.json").read_bytes())
    publish_immutable_bytes(dataset / "unpack.py", (ROOT / "src/molgap/experiment_preflight.py").read_bytes())
    pins = {"EXPECTED_PROFILE_MANIFEST_SHA256": expected_digest,
            "EXPECTED_SETUP_SHA256": sha256_file(payload / "setup.sh"),
            "EXPECTED_PROFILE_ARCHIVE_SHA256": sha256_file(dataset / "source_payload.bin"),
            "EXPECTED_UNPACK_SHA256": sha256_file(dataset / "unpack.py")}
    entry = (HERE / "kaggle_entry.py").read_text(encoding="utf-8")
    for name, value in pins.items():
        marker = name + " = None"
        if entry.count(marker) != 1:
            raise ValueError("Unrecognized diagnostic entry marker: " + name)
        entry = entry.replace(marker, name + " = " + repr(value))
    publish_immutable_bytes(kernel / "run.py", entry.encode("utf-8"))
    atomic_json(dataset / "dataset-metadata.json", {"id": DATASET,
        "title": "MolGap K1 Native T4 Profile Source V1", "licenses": [{"name": "CC0-1.0"}],
        "description": "Private immutable pure-2D train-only execution diagnostic payload; no protected roles."})
    atomic_json(kernel / "kernel-metadata.json", {"id": "nvoid912/" + RUN,
        "title": RUN, "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": []})
    receipt = {"format": "molgap-native-t4-profile-publication-binding-v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "payload_manifest_sha256": expected_digest, "dataset": DATASET,
        "requested_kernel": "nvoid912/" + RUN, "pins": pins,
        "files": {p.relative_to(output).as_posix(): sha256_file(p)
                  for p in output.rglob("*") if p.is_file()},
        "remote_action": False, "sample_decoded": False}
    atomic_json(output / "publication_binding.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.payload, args.expected_manifest_sha256, args.output), sort_keys=True))
