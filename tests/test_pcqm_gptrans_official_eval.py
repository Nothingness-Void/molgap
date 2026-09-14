import pytest
import torch
from torch_geometric.data import Data
from molgap.pcqm_gptrans_official_eval import check_part, make_evaluation_model
from molgap.training_reproducibility import configure_fp32_determinism


def test_part_rejects_changed_labels_and_identity():
    data = Data(source_idx=torch.tensor([10, 11]), y=torch.tensor([1., 2.]))
    part = {"identity": "a", "graph_sha256": "b", "source_idx": data.source_idx, "target_eV": data.y.clone(), "prediction_eV": torch.tensor([1.1, 2.2])}
    record = {"sha256": "b", "rows": 2}
    check_part(part, identity="a", record=record, data=data)
    with pytest.raises(RuntimeError):
        check_part(part, identity="c", record=record, data=data)
    part["target_eV"][0] = 0
    with pytest.raises(RuntimeError):
        check_part(part, identity="a", record=record, data=data)


def test_frozen_gptrans_can_infer_on_cpu():
    torch.set_num_threads(2)
    configure_fp32_determinism(42)
    model = make_evaluation_model().eval()
    with torch.inference_mode():
        result = model(torch.zeros(2, 9, dtype=torch.long), torch.tensor([[0, 1], [1, 0]]), torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, dtype=torch.long))
    assert result.numel() == 1 and torch.isfinite(result).all()
