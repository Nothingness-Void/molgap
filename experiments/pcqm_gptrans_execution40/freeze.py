import json
from molgap.constants import REPO_ROOT
from molgap.gptrans_execution_prepare import freeze

if __name__ == "__main__":
    print(json.dumps(freeze(REPO_ROOT)))
