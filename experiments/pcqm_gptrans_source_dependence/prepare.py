from pathlib import Path
from molgap.gptrans_source_dependence_prepare import prepare

if __name__ == "__main__":
    root=Path(__file__).resolve().parents[2]
    print(prepare(root,root/"platforms/_records/kaggle/prepared/gptrans_source_dependence_v1"))
