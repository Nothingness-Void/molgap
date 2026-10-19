"""Select one checkout in a child process without changing shared editable installs."""
import os
from pathlib import Path
import re
import subprocess
import sys

from .common.io import _safe_local


_BOOTSTRAP = """
import json, runpy, sys
source, entry, package_identity, kind = sys.argv[1:5]
sys.path.insert(0, source)
import molgap
print(json.dumps({'interpreter': sys.executable, 'molgap_module': molgap.__file__,
                  'source_root': source, 'package_identity': package_identity or None}), file=sys.stderr)
sys.argv = [entry, *sys.argv[5:]]
if kind == 'script':
    runpy.run_path(entry, run_name='__main__')
else:
    runpy.run_module(entry, run_name='__main__', alter_sys=True)
"""


def launch_local(repo_root, module, arguments=(), *, python=None, package_identity=None, scratch_root=None, script=None):
    root = Path(repo_root).absolute()
    _safe_local(root)
    source = root / "src"
    _safe_local(source)
    if not (source / "molgap/__init__.py").is_file():
        raise ValueError("Selected checkout has no MolGap package")
    if script is not None:
        from .common.paths import validate_relative_path
        from .research_memory.paths import repo_local_path
        validate_relative_path(script)
        if module is not None or not script.startswith("platforms/") or not script.endswith(".py"):
            raise ValueError("Select one module or repository-confined platform script")
        entry = repo_local_path(root, script)
        _safe_local(entry)
        if not entry.is_file():
            raise ValueError("Selected platform script is missing")
        kind = "script"
    else:
        if not isinstance(module, str) or re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module) is None:
            raise ValueError("Expected an explicit Python module")
        entry, kind = module, "module"
    interpreter = Path(python or sys.executable).absolute()
    _safe_local(interpreter)
    if package_identity is not None:
        from .common.spec import _digest
        _digest(package_identity, "package_identity")
    # -I ignores inherited PYTHONPATH and user-site state. The dependency
    # environment and its .pth files are not installed, edited or removed.
    command = [str(interpreter), "-I", "-c", _BOOTSTRAP, str(source), str(entry),
               package_identity or "", kind, *map(str, arguments)]
    environment = os.environ.copy()
    if scratch_root is not None:
        scratch = Path(scratch_root).absolute()
        _safe_local(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        environment.update(TMP=str(scratch), TEMP=str(scratch), TMPDIR=str(scratch))
    return subprocess.run(command, cwd=root, env=environment).returncode


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", type=Path)
    entry=parser.add_mutually_exclusive_group()
    entry.add_argument("--module")
    entry.add_argument("--script", help="Repository-relative platform script, using the selected checkout")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    python, paths = None, None
    if args.profile:
        from .common.paths import MachinePaths
        paths = MachinePaths.from_file(args.profile).outside_checkout(args.repo_root)
        python = paths.python
    rest = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    module = args.module or (None if args.script else "molgap.experiment_cli")
    return launch_local(args.repo_root, module, rest, python=python, script=args.script,
                        scratch_root=paths.scratch_root if paths else None)
