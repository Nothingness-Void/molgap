"""Execute a prepared single-arm local run through the registered family owner."""
from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from molgap.local_training import main
    raise SystemExit(main())
