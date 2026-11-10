"""Derive the GPTrans-only A100 notebook from the frozen two-arm builder; never execute it locally."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARM = "gptrans_pairnorm"
NAME = "MolGap_GPTrans_Full_A100.ipynb"


# 只读复制训练器的滚动检查点到输出目录之外；训练器的提交哈希链不受影响。
SNAPSHOT_FUNCTION = '''def copy_snapshot(source, directory):
    # Atomic copy; None when the trainer has not written a checkpoint yet.
    if not source.is_file():
        return None
    target = directory / ('last_checkpoint-%d.pt' % time.time_ns())
    partial = target.with_suffix('.partial')
    shutil.copyfile(source, partial)
    os.replace(partial, target)
    return target
'''
SNAPSHOT_START = '''    snapshots = RUN_ROOT / (arm + '-snapshots')
    snapshots.mkdir(exist_ok=True)
    stop_snapshots = Event()
    def snapshot_loop():
        while not stop_snapshots.wait(SNAPSHOT_SECONDS):
            try:
                print('Snapshot:', copy_snapshot(output / 'last_checkpoint.pt', snapshots), flush=True)
            except OSError as error:
                print('Snapshot skipped:', error, flush=True)
    Thread(target=snapshot_loop, daemon=True).start()
'''


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
        ("CLEANUP_SECONDS = 300\n", "CLEANUP_SECONDS = 300\nSNAPSHOT_SECONDS = 2 * 3600  # 监控快照间隔；评估只看曲线，不用于挑模型\n"),
        ("from threading import Timer\n", "from threading import Event, Thread, Timer\n" + SNAPSHOT_FUNCTION),
        ("    pause_timer.start()\n", "    pause_timer.start()\n" + SNAPSHOT_START),
        ("        pause_timer.cancel()\n", "        pause_timer.cancel()\n        stop_snapshots.set()\n"),
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
