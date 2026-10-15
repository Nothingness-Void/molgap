"""Pinned-file streaming shared by Kaggle publication and output retrieval."""
import hashlib
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping
from .experiment_launch import _safe_local
from .research_memory.paths import repo_local_path
from .research_memory.trace import file_digest


def _download_selected(*, urls: Mapping[str, str], required: Mapping[str, tuple[str, str]],
                       destination: Path, http_session=None) -> list[dict[str, str]]:
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
