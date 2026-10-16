"""Run the selected checkout using an existing dependency interpreter."""
import sys
from pathlib import Path

if __name__ == "__main__":
    # This thin entry is invoked from the selected checkout, including worktrees.
    sys.path.insert(0, str(Path.cwd() / "src"))
    from molgap.local_runtime import main
    raise SystemExit(main())
