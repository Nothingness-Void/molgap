from molgap.constants import REPO_ROOT
from molgap.gptrans_source_dependence import BASE,accept
from molgap.training_reproducibility import atomic_json

if __name__ == "__main__":
    root=REPO_ROOT
    accepted=accept(root/"platforms/_records/kaggle/training/gptrans_source_dependence_v1/gptrans_source_dependence",
                    root/"platforms/_records/kaggle/prepared/gptrans_source_dependence_v1/inputs")
    atomic_json(root/BASE/"results/acceptance_summary.json",accepted)
    print(accepted["accepted"])
