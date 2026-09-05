"""Apply the DTK 22.10 HIP version-check shim to an installed torch-scatter."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


OLD = "t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]"
NEW = "t_major, t_minor = [int(x) for x in (torch.version.cuda or torch.version.hip).split('.')[:2]]"
REQUIRED_EXTENSIONS = (
    "_scatter_cuda.so",
    "_segment_coo_cuda.so",
    "_segment_csr_cuda.so",
    "_version_cuda.so",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    package = args.site_root.resolve() / "torch_scatter"
    target = package / "__init__.py"
    text = target.read_text(encoding="utf-8")
    if OLD in text:
        temporary = target.with_name(".__init__.py.tmp")
        temporary.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
        os.replace(temporary, target)
        changed = True
    elif NEW in text:
        changed = False
    else:
        raise RuntimeError("torch-scatter version-check contract is unknown")

    extensions = {}
    for name in REQUIRED_EXTENSIONS:
        path = package / name
        if not path.is_file():
            raise RuntimeError(f"missing HIP extension: {name}")
        extensions[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    payload = {
        "format": "molgap-xian-torch-scatter-runtime-finalization-v1",
        "accepted": True,
        "changed": changed,
        "site_root": str(args.site_root.resolve()),
        "version_check": NEW,
        "extensions": extensions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
