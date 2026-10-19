"""One portable relative-path check, reused by descriptors and transport."""
from pathlib import PureWindowsPath
from typing import Any
from .spec import _text


class MachinePaths:
    """Machine-local locations, never scientific identities or role authority."""
    def __init__(self, *, python, data_root, artifact_root, scratch_root):
        from pathlib import Path
        from .io import _safe_local
        self.python = Path(python)
        self.data_root, self.artifact_root, self.scratch_root = (
            Path(p) for p in (data_root, artifact_root, scratch_root))
        for path in (self.python, self.data_root, self.artifact_root, self.scratch_root):
            _safe_local(path)
            if path == Path(path.anchor):
                raise ValueError("A drive/filesystem root is not a machine storage location")
        roots = (self.data_root, self.artifact_root, self.scratch_root)
        if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
               for i, a in enumerate(roots) for b in roots[i + 1:]):
            raise ValueError("Data, artifact and scratch roots must be separate")
        if not self.python.is_file():
            raise ValueError("Configured dependency interpreter is missing")

    @classmethod
    def from_file(cls, path):
        import json
        from pathlib import Path
        from .io import _safe_local
        from .spec import _object, _unique_object
        path = Path(path).absolute()
        _safe_local(path)
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        _object(value, "format python data_root artifact_root scratch_root", "machine profile")
        if value.pop("format") != "molgap-machine-paths-v1":
            raise ValueError("Unknown machine profile")
        return cls(**value)

    def outside_checkout(self, repo_root):
        from pathlib import Path
        from .io import _safe_local
        root = Path(repo_root).absolute()
        _safe_local(root)
        for path in (self.data_root, self.artifact_root, self.scratch_root):
            if path.is_relative_to(root) or root.is_relative_to(path):
                raise ValueError("New task storage must not overlap the selected code checkout")
        return self

    def run_directory(self, run_id, attempt_id):
        from .spec import _identifier
        from .io import _safe_local
        _identifier(run_id, "run_id")
        _identifier(attempt_id, "attempt_id")
        path = self.artifact_root / "runs" / run_id / attempt_id
        _safe_local(path)
        return path


def validate_relative_path(value: Any) -> None:
    _text(value, "descriptor path")
    # Portable repository-relative spelling avoids host-dependent drive/ADS and
    # separator interpretation; repo_local_path additionally checks symlinks.
    if "\\" in value or ":" in value or value.startswith("/"):
        raise ValueError("descriptor paths must be repository-relative POSIX paths")
    for part in value.split("/"):
        if (part in {"", ".", ".."} or part.endswith((" ", "."))
                or any(ord(c) < 32 or c in '<>"|?*' for c in part)
                or PureWindowsPath(part).is_reserved()):
            raise ValueError("invalid descriptor path or path traversal")
