"""Exercise the actual frozen CPU archive using only a synthetic WedgeData file."""
from pathlib import Path
import subprocess
import sys

from molgap.experiment_preflight import _unpack


def test_frozen_archive_contains_graph_serialization_owner(tmp_path):
    root = Path(__file__).resolve().parents[1]
    staged = root / "platforms/_records/kaggle/cycle_pair_cpu_recovery_v2/source/source_payload.bin"
    if not staged.exists():
        import pytest
        pytest.skip("Owning CPU archive is not staged on this checkout")
    package = tmp_path / "package"
    package.mkdir()
    import shutil
    shutil.copyfile(staged, package / "source.tar.gz")
    source = tmp_path / "source"
    _unpack(package, source)
    code = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import torch
from torch_geometric.data import InMemoryDataset
from molgap.pcqm_wedge import WedgeData
from molgap.packed_graph_dataset import PackedGraphDataset
import molgap.pcqm_wedge as owner
assert Path(owner.__file__).is_relative_to(Path(sys.argv[1]))
graph = WedgeData(x=torch.zeros(3,9,dtype=torch.long), edge_index=torch.tensor([[0,1],[1,0]]),
    edge_attr=torch.zeros(2,3,dtype=torch.long), source_idx=torch.tensor([0]), y=torch.tensor([1.]))
torch.save(InMemoryDataset.collate([graph]), sys.argv[2])
loaded = PackedGraphDataset(Path(sys.argv[2]), mmap=True)
assert len(loaded)==1 and loaded[0].source_idx.item()==0 and loaded[0].num_nodes==3
print("frozen WedgeData roundtrip passed")
'''
    result = subprocess.run([sys.executable, "-I", "-c", code, str(source / "src"),
        str(tmp_path / "synthetic.pt")], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
