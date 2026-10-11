from pathlib import Path
import json
from molgap.gptrans_execution_prepare import freeze

if __name__ == "__main__":
    print(json.dumps(freeze(Path(__file__).resolve().parents[2])))
