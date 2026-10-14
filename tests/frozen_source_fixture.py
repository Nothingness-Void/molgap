"""Historical source checks without substituting current scientific modules."""
import io
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile

import pytest


def stage_frozen_sources(repo, commit, destination):
    """Stage only committed MolGap sources in a fresh test-owned directory."""
    if type(commit) is not str or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Historical source fixture requires an exact Git commit")
    destination = Path(destination).resolve()
    if (destination / "src").exists():
        raise ValueError("Historical fixture source destination must be fresh")
    present = subprocess.run(["git", "cat-file", "-e", commit + "^{commit}"],
                             cwd=repo, capture_output=True)
    if present.returncode:
        pytest.skip("Recorded historical source commit is unavailable in this clone")
    result = subprocess.run(["git", "archive", "--format=tar", commit, "src/molgap"],
                            cwd=repo, capture_output=True)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            assert not name.is_absolute() and ".." not in name.parts and "\\" not in member.name
            if member.isdir():
                assert name.parts == ("src",) or name.parts[:2] == ("src", "molgap")
                continue
            assert name.parts[:2] == ("src", "molgap") and member.isfile() and not member.linkname
            target = destination / member.name
            assert target.resolve().is_relative_to(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    return destination


def run_frozen_source(snapshot, script, *args):
    """Run trusted test assertions in an isolated interpreter with frozen imports."""
    bootstrap = """import sys
from pathlib import Path
snapshot = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(snapshot / 'src'))
import molgap
assert Path(molgap.__file__).resolve().is_relative_to(snapshot)
"""
    return subprocess.run([sys.executable, "-I", "-c", bootstrap + script, str(snapshot),
        *(str(value) for value in args)], cwd=snapshot, capture_output=True, text=True)
