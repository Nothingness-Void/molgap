"""One portable relative-path check, reused by descriptors and transport."""
from pathlib import PureWindowsPath
from typing import Any
from .spec import _text


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
