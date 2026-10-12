"""Compose existing execution-assay declarations; no training or submission."""
import json
from pathlib import Path

from molgap.gptrans_execution_prepare import freeze
from molgap.gptrans_execution_tuning import ExecutionTuning
from molgap.training_reproducibility import atomic_json

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    base = "experiments/pcqm_gptrans_execution_tuning40"
    metadata = json.loads((root / "experiments/pcqm_gptrans_execution40/gpu/kernel-metadata.json").read_text())
    metadata.update(id="kaseichou/molgap-gptrans-tuning40-dual-s42", title="MolGap GPTrans Tuning40 Dual S42",
                    dataset_sources=["kaseichou/molgap-gptrans-tuning40-source", "kaseichou/pcqm4mv2-ogb-fixed-100k-v1"])
    atomic_json(root / base / "gpu/kernel-metadata.json", metadata)
    print(json.dumps(freeze(root, base=base, run="gptrans-tuning40-dual-s42",
                           tuning=ExecutionTuning(independent_models=1, total_cpu_budget=2))))
