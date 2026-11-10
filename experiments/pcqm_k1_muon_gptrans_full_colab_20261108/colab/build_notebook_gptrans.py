"""Derive the GPTrans-only A100 notebook from the frozen two-arm builder; never execute it locally."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARM = "gptrans_pairnorm"
NAME = "MolGap_GPTrans_Full_A100.ipynb"


def patched_source():
    text = (HERE / "build_notebook.py").read_text(encoding="utf-8")
    edits = [
        ("ARM_ORDER = ('k1_muon', 'gptrans_pairnorm')", f"ARM_ORDER = ('{ARM}',)"),
        ("'k1_muon/package'", f"'{ARM}/package'"),
        ("MolGap_K1_Muon_GPTrans_Full_A100.ipynb", NAME),
        ("releases = json.loads((HERE / 'release_assets.json').read_text())",
         f"releases = [row for row in json.loads((HERE / 'release_assets.json').read_text()) if row['arm_id'] == '{ARM}']"),
        ("# MolGap：加速 K1＋GPTrans 全量 Colab", "# MolGap：GPTrans BondLocal＋PairNorm 全量 Colab"),
        ("2. 单张 A100 顺序训练加速 K1 Muon、GPTrans BondLocal＋PairNorm，共享 SSD 数据。",
         "2. 单张 A100 训练 GPTrans BondLocal＋PairNorm（B128，冻结配方）。"),
        ("K1完成后继续GPTrans。", ""),
    ]
    for old, new in edits:
        assert text.count(old) >= 1, old
        text = text.replace(old, new)
    return text


def main():
    exec(compile(patched_source(), str(HERE / "build_notebook.py"), "exec"), {"__file__": str(HERE / "build_notebook.py"), "__name__": "__main__"})
    path = HERE / NAME
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert [c["id"] for c in notebook["cells"]].count("a100-run") == 1
    print(NAME, len(notebook["cells"]), "cells")


if __name__ == "__main__":
    main()
