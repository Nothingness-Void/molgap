"""Launch the frozen GPTrans flow ablations through the shared runtime."""
import json
import os
import shutil
import sys
from pathlib import Path


def _install_source_path():
    try:
        import molgap  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    archives = list(Path("/kaggle/input").rglob("source_payload.bin"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one source payload, found {archives}")
    expanded = Path("/kaggle/working/_flow_source")
    if expanded.exists():
        shutil.rmtree(expanded)
    shutil.unpack_archive(archives[0], expanded, format="gztar")
    sys.path.insert(0, str(expanded / "src"))

os.environ.setdefault(
    "MOLGAP_SCREEN_MODES",
    json.dumps(["no_pair_to_node", "no_pair_recurrence"]),
)
os.environ.setdefault(
    "MOLGAP_SCREEN_ROOT", "/kaggle/working/gptrans_flow_ablation"
)
os.environ.setdefault("MOLGAP_PLATFORM_ID", "kaggle1-t4")

_install_source_path()

from molgap.gptrans_kaggle_runtime import main


if __name__ == "__main__":
    main()
