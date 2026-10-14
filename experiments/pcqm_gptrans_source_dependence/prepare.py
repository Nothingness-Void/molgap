from molgap.constants import REPO_ROOT
from molgap.gptrans_source_dependence_prepare import prepare

if __name__ == "__main__":
    root=REPO_ROOT
    print(prepare(root,root/"platforms/_records/kaggle/prepared/gptrans_source_dependence_v1"))
