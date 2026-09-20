"""Launch the two zero-initialized conditional GPTrans candidates."""
import json
import os
import shutil
import sys
from pathlib import Path


archives = list(Path("/kaggle/input").rglob("source_payload.bin"))
if len(archives) != 1:
    raise RuntimeError(f"Expected one source payload, found {archives}")
expanded = Path("/kaggle/working/_conditional_flow_source")
if expanded.exists():
    shutil.rmtree(expanded)
shutil.unpack_archive(archives[0], expanded, format="gztar")
sys.path.insert(0, str(expanded / "src"))

os.environ.setdefault(
    "MOLGAP_SCREEN_MODES",
    json.dumps(["conditional_pair_readback", "conditional_pair_recurrence"]),
)
os.environ.setdefault(
    "MOLGAP_SCREEN_ROOT", "/kaggle/working/gptrans_conditional_flow"
)
os.environ.setdefault("MOLGAP_PLATFORM_ID", "kaggle1-t4x2")

from molgap.gptrans_kaggle_runtime import main


if __name__ == "__main__":
    main()
