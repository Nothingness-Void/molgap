"""Install a P100-compatible PyTorch wheel and verify CUDA execution on Kaggle."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            "--force-reinstall",
            "torch==2.7.1+cu126",
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
        ],
        check=True,
    )

    import torch

    assert torch.cuda.is_available(), "Kaggle GPU was not attached"
    assert "sm_60" in torch.cuda.get_arch_list(), "Installed torch does not support P100 sm_60"
    probe = torch.ones(1024, 1024, device="cuda") @ torch.ones(1024, 1024, device="cuda")
    record = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "architectures": torch.cuda.get_arch_list(),
        "probe_sum": float(probe.sum().item()),
    }
    Path("/kaggle/working/p100_torch_probe.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
