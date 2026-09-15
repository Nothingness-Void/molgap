"""Check the deployed archive, including non-Python runtime dependencies."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile


def test_packaged_fusion_contract_is_present_and_hash_bound(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "repair_package", root / "experiments/pcqm_gptrans_full_convergence/package_repair.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Read real source files, but direct generated artifacts to a temporary tree.
    class StagedRoot:
        def __truediv__(self, name):
            return (tmp_path if str(name).startswith('platforms/_records') else root) / name

        def __fspath__(self):
            return str(root)

    monkeypatch.setattr(module, "ROOT", StagedRoot())
    module.main()
    with tarfile.open(tmp_path / "platforms/_records/ims/convergence_repair_r4/runtime.tar.gz") as archive:
        inventory = json.load(archive.extractfile("code/runtime_files.json"))
        name = "experiments/pcqm_k1_gptrans_full_fusion/fusion_contract.json"
        data = archive.extractfile("code/" + name).read()
        assert hashlib.sha256(data).hexdigest() == inventory[name]
        assert json.loads(data) == json.loads((root / name).read_text())
        for name, expected in inventory.items():
            assert hashlib.sha256(archive.extractfile("code/" + name).read()).hexdigest() == expected
