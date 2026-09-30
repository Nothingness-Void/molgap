"""Thin saved-output GPU acceptance entry; no model execution or remote action."""
import argparse
from pathlib import Path
from molgap.gptrans_author_acceptance import accept_training_outputs
from molgap.research_memory.trace import atomic_write, json_bytes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    result = accept_training_outputs(Path(__file__).resolve().parents[2], args.records, args.package)
    atomic_write(args.output, json_bytes(result))
    print({"accepted": result["accepted"], "arms": {k: v["material_gain_eV"] for k,v in result["arms"].items()}})
    if args.finalize:
        from molgap.gptrans_author_terminal import close_author_outputs
        print(close_author_outputs(Path(__file__).resolve().parents[2], args.records, args.output))
