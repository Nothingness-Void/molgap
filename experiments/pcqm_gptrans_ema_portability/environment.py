"""CPU-only qualification of the identical hash-bound GPU runtime bootstrap."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tarfile


def main():
    matches = list(Path("/kaggle/input").rglob("audit_release.json"))
    if len(matches) != 1:
        raise ValueError("One frozen source release required")
    inputs = matches[0].parent
    release = json.loads(matches[0].read_text())
    archive = inputs/"source_payload.bin"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != release["archive_sha256"]:
        raise ValueError("Source payload changed")
    with tarfile.open(archive) as source:
        payload = source.extractfile("experiments/pcqm_gptrans_ema_portability/run.py").read()
    if hashlib.sha256(payload).hexdigest() != release["entry_sha256"]:
        raise ValueError("Qualification must use the exact audit entry")
    entry = Path("/kaggle/working/qualified_audit_entry.py")
    entry.write_bytes(payload)
    subprocess.run([sys.executable, str(entry), "--environment-only"], check=True, timeout=1850)


if __name__ == "__main__":
    main()
