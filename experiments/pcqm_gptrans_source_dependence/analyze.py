from pathlib import Path
from molgap.gptrans_source_dependence import BASE
from molgap.gptrans_source_dependence_analysis import analyze
from molgap.training_reproducibility import atomic_json

if __name__=="__main__":
    root=Path(__file__).resolve().parents[2]
    result=analyze(root/"platforms/_records/kaggle/training/gptrans_source_dependence_v1/gptrans_source_dependence",
                   root/"platforms/_records/kaggle/prepared/gptrans_source_dependence_v1/inputs")
    atomic_json(root/BASE/"results/descriptive_analysis.json",result)
    print("descriptive_saved_observation_analysis_complete")
