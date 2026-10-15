"""SHA-pinned streaming and atomic retention; not scientific acceptance."""
import hashlib
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping
from .io import _safe_local
from .spec import _digest
from .paths import validate_relative_path as _path
from ..research_memory.paths import repo_local_path
from ..research_memory.trace import file_digest


def download_pinned_files(*, urls: Mapping[str, str], required: Mapping[str, tuple[str, str]],
                       destination: Path, http_session=None) -> list[dict[str, str]]:
    if not isinstance(required, Mapping) or not required:
        raise ValueError("Pinned downloads require a non-empty allowlist")
    paths = set()
    for remote, binding in required.items():
        _path(remote)
        if not isinstance(binding, (tuple, list)) or len(binding) != 2:
            raise ValueError("Pinned download requires a local path and SHA256")
        relative, pin = binding
        _path(relative)
        _digest(pin, "download SHA256")
        if relative.casefold() in paths:
            raise ValueError("Pinned download paths are aliased")
        paths.add(relative.casefold())
    if not isinstance(urls, Mapping) or set(urls) != set(required):
        raise ValueError("Download URLs must exactly match the pinned allowlist")
    root = Path(destination).absolute()
    _safe_local(root)
    root.mkdir(parents=True, exist_ok=True)
    if http_session is None:
        import requests

        http_session = requests.Session()
    retained = []
    for remote, (relative, pin) in required.items():
        # Check the lexical path before resolving it, so a reparse point cannot
        # redirect an output outside the destination between these operations.
        target = root / relative
        _safe_local(target)
        target = repo_local_path(root, relative)
        if target.exists():
            if not target.is_file() or file_digest(target) != pin:
                raise ValueError("Retained output conflicts with pinned artifact")
            retained.append({"path": relative, "sha256": pin, "state": "ALREADY_RETAINED"})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            digest = hashlib.sha256()
            try:
                with http_session.get(urls[remote], stream=True, timeout=(30, 60)) as response:
                    if response.status_code != 200:
                        raise RuntimeError("Selected file retrieval returned a non-200 response")
                    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                        temporary = Path(handle.name)
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                digest.update(chunk)
                                handle.write(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
            except Exception:
                # Do not expose a signed URL from an SDK/HTTP exception.
                raise RuntimeError("Selected output retrieval failed") from None
            if digest.hexdigest() != pin:
                raise ValueError("Retrieved output hash disagrees with pinned attempt")
            _safe_local(target)
            try:
                os.link(temporary, target)
            except FileExistsError:
                if not target.is_file() or file_digest(target) != pin:
                    raise ValueError("Concurrent output publication conflicts")
        except ValueError:
            raise
        except Exception:
            raise RuntimeError("Selected output retrieval failed") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        retained.append({"path": relative, "sha256": pin, "state": "RETAINED"})
    return retained
