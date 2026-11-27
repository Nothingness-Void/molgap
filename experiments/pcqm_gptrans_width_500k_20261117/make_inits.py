"""Build the GPT-T initial states of the width/FFN/depth 500K question and print their identities.

Run with the repo venv and PYTHONPATH=<checkout>/src. Existing files (R s42, R s43, L20 s42) are checked, not rebuilt:
their state sha must equal a fresh build of the same shape and seed. New files are written once and never overwritten.
Usage: make_inits.py OUT_DIR RECORD.json
"""
import json
import os
import sys
from pathlib import Path

import torch

from molgap.gptrans_bond_local import make_bond_local_model
from molgap.training_reproducibility import sha256_file
from molgap.v4_runtime import read_frozen_state_artifact, state_dict_sha256

# arm -> (LAYERS, WIDTH, HEADS, FFN_RATIO, expected parameter count)
SHAPES = {"R": (12, 256, 8, 1, 5_871_201), "L20": (20, 256, 8, 1, 9_529_761),
          "W320": (12, 320, 10, 1, None), "F3": (12, 256, 8, 3, None)}
GDRIVE = Path("G:/我的云端硬盘/MolGap/code")
EXISTING = {("R", 42): GDRIVE / "gptrans_init.pt", ("R", 43): GDRIVE / "gptrans_init_s43.pt",
            ("L20", 42): Path("D:/molgap-artifacts/initializations/gptrans-l20-levers-20261115/gptrans_init_l20_s42.pt")}
FILE_NAME = {("R", 42): "gptrans_init.pt", ("R", 43): "gptrans_init_s43.pt"}


def build(arm, seed):
    layers, width, heads, ffn, _ = SHAPES[arm]
    os.environ.update(MOLGAP_GPTRANS_LAYERS=str(layers), MOLGAP_GPTRANS_WIDTH=str(width),
                      MOLGAP_GPTRANS_HEADS=str(heads), MOLGAP_GPTRANS_FFN_RATIO=str(ffn))
    model = make_bond_local_model(seed)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, sum(p.numel() for p in model.parameters())


def main(out_dir, record_path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert state_dict_sha256(build("R", 42)[0]) == "a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b", "seed-42 R build lost"
    record = {}
    for arm in SHAPES:
        for seed in (42, 43):
            state, params = build(arm, seed)
            sha = state_dict_sha256(state)
            want = SHAPES[arm][4]
            assert want is None or params == want, (arm, params)
            path = EXISTING.get((arm, seed)) or out_dir / f"gptrans_init_{arm.lower()}_s{seed}.pt"
            if not path.exists():
                torch.save({"model_state": state, "state_sha256": sha}, path)
            _, info = read_frozen_state_artifact(path, expected_state_sha256=sha)
            record[f"{arm}-s{seed}"] = {"init_file": FILE_NAME.get((arm, seed), f"gptrans_init_{arm.lower()}_s{seed}.pt"), "path": path.as_posix(),
                                        "file_sha256": sha256_file(path), "state_sha256": sha, "params": params,
                                        "shape": dict(zip(("layers", "width", "heads", "ffn_ratio"), SHAPES[arm][:4])), "torch_seed": seed}
            print(arm, seed, params, sha, record[f"{arm}-s{seed}"]["file_sha256"], flush=True)
    Path(record_path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:3])
