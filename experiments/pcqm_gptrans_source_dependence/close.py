from molgap.constants import REPO_ROOT
from molgap.gptrans_source_dependence_records import close

if __name__=="__main__":
    root=REPO_ROOT
    print(close(root,root/"platforms/_records/kaggle/training/gptrans_source_dependence_v1/gptrans_source_dependence",
                root/"platforms/_records/kaggle/prepared/gptrans_source_dependence_v1/inputs"))
