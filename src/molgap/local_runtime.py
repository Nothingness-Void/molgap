"""Select one checkout in a child process without changing shared editable installs."""
import os
from pathlib import Path
import re
import subprocess
import sys

from .common.io import _safe_local


_BOOTSTRAP = """
import json, runpy, sys
source, module, package_identity = sys.argv[1:4]
sys.path.insert(0, source)
import molgap
print(json.dumps({'interpreter': sys.executable, 'molgap_module': molgap.__file__,
                  'source_root': source, 'package_identity': package_identity or None}), file=sys.stderr)
sys.argv = [module, *sys.argv[4:]]
runpy.run_module(module, run_name='__main__', alter_sys=True)
"""


def launch_local(repo_root, module, arguments=(), *, python=None, package_identity=None):
    root = Path(repo_root).absolute()
    _safe_local(root)
    source = root / "src"
    _safe_local(source)
    if not (source / "molgap/__init__.py").is_file():
        raise ValueError("Selected checkout has no MolGap package")
    if not isinstance(module, str) or re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module) is None:
        raise ValueError("Expected an explicit Python module")
    interpreter = Path(python or sys.executable).absolute()
    _safe_local(interpreter)
    if package_identity is not None:
        from .common.spec import _digest
        _digest(package_identity, "package_identity")
    # -I ignores inherited PYTHONPATH and user-site state. The dependency
    # environment and its .pth files are not installed, edited or removed.
    command = [str(interpreter), "-I", "-c", _BOOTSTRAP, str(source), module,
               package_identity or "", *map(str, arguments)]
    return subprocess.run(command, cwd=root, env=os.environ.copy()).returncode


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--module", default="molgap.experiment_cli")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    python = None
    if args.profile:
        from .common.paths import MachinePaths
        python = MachinePaths.from_file(args.profile).outside_checkout(args.repo_root).python
    rest = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    return launch_local(args.repo_root, args.module, rest, python=python)
