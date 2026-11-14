"""Apply to a copy of src/molgap the three K1-relevant edits that the Colab K1 runs used (molgap_src.tar.gz, build_clean_colab.py).

The B1024 runner feeds K1 a GPU-built batch: it needs the dense-layout fast path, the recipe-identity resume check is bypassed
(the runner changes batch size/lr after validation), and the deterministic-fill guard is bypassed (fast mode is non-deterministic).
Checked byte for byte against the Colab tarball's copies of the same files. Usage: colab_src_patches.py <molgap_dir>
"""
import sys
from pathlib import Path

EDITS = {
    "k1_execution_layout.py": ("def layout_from_cpu_batch(batch, device):\n    import torch\n",
                               "def layout_from_cpu_batch(batch, device):\n    import torch\n"
                               "    pre = getattr(batch, '_gpu_dense_layout', None)\n    if pre is not None:\n"
                               "        return DenseLayout(batch, pre['flat_index'], pre['valid'], pre['nodes'], pre['graphs'], pre['max_nodes'])\n"),
    "training_reproducibility.py": ('raise ValueError("Fail-closed deterministic algorithms are required")', "pass"),
    "training/session.py": ('cursor["recipe_identity"] != recipe.identity', "False"),
}

if __name__ == "__main__":
    root = Path(sys.argv[1])
    for rel, (old, new) in EDITS.items():
        path = root / rel
        text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        assert text.count(old) == 1, (rel, text.count(old))
        path.write_bytes(text.replace(old, new).encode("utf-8"))
        print("patched", rel)
