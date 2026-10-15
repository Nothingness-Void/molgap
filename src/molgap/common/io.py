"""Local path safety and immutable publication, without launch/controller imports."""
import os
from pathlib import Path
import stat
import tempfile


def _safe_local(path):
    if type(path) is not type(Path()) or not path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected absolute concrete local Path without traversal")
    if str(path).startswith(("\\\\", "//")):
        raise ValueError("Network paths are forbidden")
    for part in (path, *path.parents):
        if part != Path(part.anchor) and (
                ":" in part.name or part.name.endswith((".", " "))
                or any(ord(c) < 32 for c in part.name)
                or part.name.split(".")[0].upper() in {
                    "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)),
                    *(f"LPT{i}" for i in range(10))}):
            raise ValueError("Unsafe local path component")
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/junction/reparse paths are forbidden")


def publish_immutable_bytes(path: Path, payload: bytes) -> Path:
    """Share the receipt's atomic no-overwrite publication with producer metadata."""
    _safe_local(path)
    if not isinstance(payload, bytes):
        raise TypeError("Immutable publication requires exact bytes")
    output_dir = path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    def identical():
        _safe_local(path)
        if not stat.S_ISREG(path.stat().st_mode) or path.read_bytes() != payload:
            raise ValueError("Receipt retry conflict; refusing overwrite")

    if path.exists():
        identical()
        return path
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".launch-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _safe_local(output_dir)
        try:
            os.link(temporary, path)
        except FileExistsError:
            identical()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
