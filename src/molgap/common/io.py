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


def copy_immutable_file(source: Path, destination: Path, *, expected_sha256: str) -> dict:
    """Copy then stream-verify; never delete the source or clobber another copy."""
    from .spec import _digest
    from ..research_memory.trace import file_digest
    _digest(expected_sha256, "retained artifact SHA256")
    source, destination = Path(source).absolute(), Path(destination).absolute()
    _safe_local(source)
    _safe_local(destination)
    if not stat.S_ISREG(source.stat().st_mode) or file_digest(source) != expected_sha256:
        raise ValueError("Retained source identity mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not stat.S_ISREG(destination.stat().st_mode) or file_digest(destination) != expected_sha256:
            raise ValueError("Retained artifact retry conflict")
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".copy-", delete=False) as handle:
                temporary = Path(handle.name)
                with source.open("rb") as reader:
                    while block := reader.read(1024 * 1024):
                        handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
            if file_digest(temporary) != expected_sha256:
                raise ValueError("Copied artifact checksum mismatch; original retained")
            _safe_local(destination)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if file_digest(destination) != expected_sha256:
                    raise ValueError("Concurrent retained artifact conflict")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if file_digest(source) != expected_sha256 or file_digest(destination) != expected_sha256:
        raise ValueError("Retained artifact changed during relocation")
    return {"path": str(destination), "sha256": expected_sha256,
            "bytes": destination.stat().st_size, "original_retained": True}
