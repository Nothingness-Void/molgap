"""Bootstrap the committed shared runtime; model and orchestration live in src."""
import json
import os
import shutil
import sys
from pathlib import Path

if __name__ == "__main__":
    modules = list(Path("/kaggle/input").rglob("src/molgap/gptrans_kaggle_runtime.py"))
    if len(modules) == 1:
        root = modules[0].parents[1]
    else:
        archives = list(Path("/kaggle/input").rglob("source_payload.bin"))
        if len(archives) != 1:
            raise RuntimeError("Ambiguous or missing source payload")
        expanded = Path("/kaggle/working/_memory_bootstrap")
        shutil.unpack_archive(archives[0], expanded, format="gztar")
        root = expanded / "src"
    sys.path.insert(0, str(root))
    os.environ["MOLGAP_SCREEN_MODES"] = json.dumps(["memory_value", "memory_message"])
    os.environ["MOLGAP_SCREEN_ROOT"] = "/kaggle/working/gptrans_memory_readback"
    from molgap.gptrans_kaggle_runtime import main
    main()
