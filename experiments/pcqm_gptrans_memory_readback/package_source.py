"""Reuse source-only packaging without constructing local models."""
import importlib.util
from molgap.constants import REPO_ROOT

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("pack", REPO_ROOT / "experiments/pcqm_gptrans_relation_flow/package_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(experiment="pcqm_gptrans_memory_readback", dataset_id="kaseichou/molgap-gptrans-memory-readback-source", title="MolGap GPTrans Memory Readback Source")
