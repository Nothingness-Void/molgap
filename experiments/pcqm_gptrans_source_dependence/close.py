from pathlib import Path
from molgap.gptrans_source_dependence_records import close

if __name__=="__main__":
    root=Path(__file__).resolve().parents[2]
    print(close(root,root/"platforms/_records/kaggle/training/gptrans_source_dependence_v1/gptrans_source_dependence",
                root/"platforms/_records/kaggle/prepared/gptrans_source_dependence_v1/inputs"))
