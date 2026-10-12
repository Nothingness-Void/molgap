"""Small synthetic input/backend checks; no retained models or datasets."""
import ast
from pathlib import Path
from types import SimpleNamespace, MethodType

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch

from molgap.gptrans_execution_backend import (prepare_layout, dense_inputs, EpochBatches,
    buffered_evaluate, assert_static_dataset)


def batch():
    return Batch.from_data_list([
        Data(x=torch.tensor([[1], [2]]), edge_index=torch.tensor([[0, 1], [1, 0]]),
             edge_attr=torch.tensor([[1], [1]]), y=torch.tensor([.2]), source_idx=torch.tensor([100000])),
        Data(x=torch.tensor([[3], [4], [5]]), edge_index=torch.tensor([[0, 1], [1, 0]]),
             edge_attr=torch.tensor([[2], [2]]), y=torch.tensor([.3]), source_idx=torch.tensor([100001]))])


def test_layout_matches_pyg_and_edges():
    b = prepare_layout(batch())
    layout = b.execution_layout
    dense, mask = to_dense_batch(b.x, b.batch)
    result = b.x.new_zeros((layout['batch_size'] * layout['max_nodes'], 1))
    result[layout['flat_index']] = b.x
    assert torch.equal(result.view_as(dense), dense)
    assert torch.equal(layout['mask'], mask)
    assert tuple(x.tolist() for x in layout['edges']) == ([0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0])


def test_layout_rejects_intergraph_bond():
    b = batch(); b.edge_index[1, 0] = 2
    with pytest.raises(ValueError, match='crosses'):
        prepare_layout(b)


def test_layout_rejects_unordered_membership():
    b = batch(); b.batch[0] = 1
    with pytest.raises(ValueError, match='membership'):
        prepare_layout(b)


def test_epoch_sampler_preserves_frozen_order():
    from molgap import pcqm_gptrans_v4 as native
    sampler = EpochBatches(native, native.TRAIN_ROWS)
    for epoch in (0, 1, 39):
        sampler.epoch = epoch
        assert list(sampler) == list(native.DeterministicEpochBatchSampler(native.TRAIN_ROWS, epoch))


def test_static_dataset_rejects_transform():
    with pytest.raises(ValueError, match='transforms'):
        assert_static_dataset(SimpleNamespace(transform=lambda x: x))


def test_foreach_ema_respects_own_decay_and_reload():
    from test_gptrans_execution_hotpath import ScalarEMA
    from molgap.pcqm_gptrans_v4 import ExponentialMovingAverage
    # An elementary CPU layer tests arithmetic only; no MolGap model executes.
    layer = torch.nn.Linear(2, 1)
    for decay in (.999, .9999):
        base, fast = ScalarEMA(layer, decay), ExponentialMovingAverage(layer, decay)
        for index in range(3):
            with torch.no_grad():
                layer.weight.add_(.001)
            base.update(layer); fast.update(layer)
            for key in base.state:
                assert torch.equal(base.state[key], fast.state[key])
        fast.load_state_dict(base.state_dict())
        base.update(layer); fast.update(layer)
        assert all(torch.equal(base.state[key], fast.state[key]) for key in base.state)


def test_execution_ema_import_does_not_require_k1_profile():
    import subprocess
    import sys

    # A fresh interpreter prevents an earlier test import masking the missing mount.
    code = (
        "import sys; "
        "sys.modules['molgap.pcqm_k1_pair_token_batch_profile'] = None; "
        "sys.modules['molgap.pcqm_gptrans_step_profile'] = None; "
        "from molgap.gptrans_execution_hotpath import BoundForeachEMA; "
        "from molgap.gptrans_execution_backend import backend; "
        "assert callable(BoundForeachEMA.update)"
    )
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_buffered_cpu_evaluation_preserves_payload_and_live_weights():
    from molgap.pcqm_gptrans_v4 import ExponentialMovingAverage
    model = torch.nn.Linear(1, 1)
    ema = ExponentialMovingAverage(model)
    initial = {k:v.clone() for k,v in model.state_dict().items()}
    b = prepare_layout(batch())
    result = buffered_evaluate(SimpleNamespace(DEVELOPMENT_ROWS=2),
        lambda model, batch: batch.y.view(-1) + .01, lambda graphs:[b],
        model, None, torch.tensor(0.), torch.tensor(1.), device='cpu')
    assert result['source_idx'].tolist() == [100000, 100001]
    assert result['mae_eV'] == float((result['prediction_eV'] - result['target_eV']).abs().mean())
    assert all(torch.equal(initial[k], model.state_dict()[k]) for k in initial)


def test_optimized_input_assembly_has_no_gpu_scalar_or_mask_equality():
    source = Path('src/molgap/gptrans_execution_backend.py').read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name=='dense_inputs')
    text = ast.get_source_segment(source,node)
    assert '.item(' not in text and 'torch.equal' not in text and 'to_dense_batch' not in text


def test_instrumentation_restores_native_methods():
    from molgap.gptrans_execution_backend import backend
    events = []
    owner = SimpleNamespace(_optimizer_step=lambda: events.append("step"),
                            _save_checkpoint=lambda: None)
    original = owner._optimizer_step
    with backend(owner) as report:
        owner._optimizer_step()
        assert report["training_batches"] == 1
    assert events == ["step"] and owner._optimizer_step is original


def test_native_default_ema_equation_is_unchanged():
    from molgap.pcqm_gptrans_v4 import ExponentialMovingAverage, EMA_DECAY
    model = torch.nn.Linear(2, 1)
    ema = ExponentialMovingAverage(model)
    expected = {k: v.clone() for k, v in ema.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(.01)
        model.bias.add_(.02)
    for k, v in model.state_dict().items():
        expected[k].mul_(EMA_DECAY).add_(v.detach(), alpha=1 - EMA_DECAY)
    ema.update(model)
    assert ema.decay == EMA_DECAY
    assert all(torch.equal(expected[k], v) for k, v in ema.state_dict().items())


@pytest.mark.parametrize('decay', [.998, 0, 1, True, float('nan')])
def test_native_ema_rejects_unreviewed_decay(decay):
    from molgap.pcqm_gptrans_v4 import ExponentialMovingAverage
    with pytest.raises(ValueError, match='reviewed frozen'):
        ExponentialMovingAverage(torch.nn.Linear(1, 1), decay)
