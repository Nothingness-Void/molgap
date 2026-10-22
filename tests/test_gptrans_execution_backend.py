"""Small synthetic input/backend checks; no retained models or datasets."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch

from molgap.gptrans_execution_backend import (prepare_layout,
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


def test_static_dataset_rejects_transform():
    with pytest.raises(ValueError, match='transforms'):
        assert_static_dataset(SimpleNamespace(transform=lambda x: x))


def test_foreach_ema_respects_own_decay_and_reload():
    from molgap.pcqm_gptrans_step_profile import ForeachExponentialMovingAverage
    from molgap.pcqm_gptrans_v4 import ExponentialMovingAverage
    # An elementary CPU layer tests arithmetic only; no MolGap model executes.
    layer = torch.nn.Linear(2, 1)
    for decay in (.999, .9999):
        base, fast = ExponentialMovingAverage(layer, decay), ForeachExponentialMovingAverage(layer, decay)
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
        "from molgap.pcqm_gptrans_step_profile import ForeachExponentialMovingAverage; "
        "from molgap.gptrans_execution_backend import backend; "
        "assert callable(ForeachExponentialMovingAverage.update)"
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
        model, ema, None, torch.tensor(0.), torch.tensor(1.), device='cpu')
    assert result['source_idx'].tolist() == [100000, 100001]
    assert result['mae_eV'] == float((result['prediction_eV'] - result['target_eV']).abs().mean())
    assert all(torch.equal(initial[k], model.state_dict()[k]) for k in initial)


def test_persistent_dev_creation_preserves_original_main_rng_draw():
    from torch_geometric.loader import DataLoader
    from molgap.gptrans_execution_backend import backend
    names = ('_make_model', '_training_loader', '_development_loader', '_forward',
             '_make_training_state', '_optimizer_step', '_evaluate', '_save_checkpoint', '_scientific_fields')
    owner = SimpleNamespace(**{name: lambda *a, **k: None for name in names}, PHYSICAL_BATCH=128)
    graphs = [Data(x=torch.ones(1,1))]
    torch.manual_seed(420)
    next(iter(DataLoader(graphs, num_workers=0)))
    expected = torch.get_rng_state()
    torch.manual_seed(420)
    with backend(owner, True):
        owner._development_loader(graphs)
        assert torch.equal(torch.get_rng_state(), expected)


def test_execution_source_does_not_assume_published_plan_mount():
    source = Path('src/molgap/gptrans_execution_study.py').read_text()
    assert 'plan_spec_sha256' in source
    assert '/ "prospective"' not in source
    assert '(arm + "_plan.json")' in source


def test_execution_retention_rejects_missing_prefix_before_tensor_loading(tmp_path):
    from molgap.gptrans_execution_acceptance import verified_files
    with pytest.raises(ValueError, match='retention'):
        verified_files(tmp_path, {'files':{}})


@pytest.mark.parametrize('arm', ['execution_original', 'execution_optimized'])
def test_acceptance_fields_match_declared_execution_backend(arm):
    from molgap.gptrans_execution_acceptance import expected_execution_fields, validate_execution_fields
    fields = expected_execution_fields(arm)
    validate_execution_fields(arm, fields)
    assert fields['precision'] == 'fp32' and fields['seed'] == 42
    assert fields['execution_backend'] == ('optimized-v1' if arm.endswith('optimized') else 'original-v1')
    # This field belongs to the native horizon; actual40 exposure is separately checked.
    assert fields['sample_exposure'] == 5998080


@pytest.mark.parametrize('key,value', [
    ('precision', 'bf16'), ('seed', 43), ('target_fingerprint', 'another-target'),
    ('optimizer_fingerprint', '0' * 64), ('schedule_fingerprint', '0' * 64),
    ('execution_backend', 'original-v1'), ('sample_exposure', 3998720),
])
def test_acceptance_rejects_self_consistent_but_unfrozen_fields(key, value):
    from molgap.gptrans_execution_acceptance import expected_execution_fields, validate_execution_fields
    observed = expected_execution_fields('execution_optimized')
    observed[key] = value
    with pytest.raises(ValueError, match='Frozen scientific fields'):
        validate_execution_fields('execution_optimized', observed)


@pytest.mark.parametrize('key,value', [('physical_batch_per_device', 64), ('tf32_enabled', True)])
def test_runtime_gate_rejects_batch_or_tf32_drift_even_with_rehashed_certificate(key, value):
    from molgap.screen_policy import canonical_fingerprint, validate_runtime_certificate
    from molgap.gptrans_execution_acceptance import expected_execution_fields
    certificate = {
        'format': 'molgap-runtime-certificate-v1', 'status': 'accepted',
        'platform_id': 'synthetic', 'accelerator': 'Tesla T4', 'precision': 'fp32',
        'tf32_enabled': False, 'deterministic_algorithms': True,
        'physical_batch_per_device': 128, 'tail_batch_policy': 'drop_last',
        'calibration_checks_passed': True,
        **{k: 'a' * 64 for k in ('software_fingerprint', 'determinism_fingerprint',
            'calibration_fixture_sha256', 'calibration_output_sha256', 'runtime_fingerprint')},
    }
    contract = {**expected_execution_fields('execution_original'),
        'platform_id': 'synthetic', 'accelerator': 'Tesla T4',
        'runtime_certificate_id': canonical_fingerprint(certificate)}
    validate_runtime_certificate(certificate, contract)
    certificate[key] = value
    contract['runtime_certificate_id'] = canonical_fingerprint(certificate)
    with pytest.raises(ValueError, match='Runtime certificate mismatch'):
        validate_runtime_certificate(certificate, contract)


def test_packaged_plan_uses_planner_defaults_before_validation():
    import json
    from molgap.research_memory.plan import _prepare_trajectory
    from molgap.research_memory.schemas import validate_trajectory
    for arm in ('execution_original','execution_optimized'):
        path = Path('experiments/pcqm_gptrans_execution40/gpu') / arm / 'plan_input.json'
        plan = _prepare_trajectory(json.loads(path.read_text()))
        validate_trajectory(plan)
    source = Path('src/molgap/gptrans_execution_study.py').read_text()
    assert '_prepare_trajectory(json.loads(staged.read_text()))' in source


def test_execution_timing_report_preserves_scopes_and_observed_counters():
    from molgap.gptrans_execution_acceptance import timing_summary
    rows = [{'elapsed_seconds': 10. + i} for i in range(40)]
    phases = dict(optimizer_call_seconds=20., evaluation_seconds=80.,
        checkpoint_publication_seconds=40., training_batches=31240, evaluation_passes=80)
    result = timing_summary(rows, 2000., phases)
    assert result['first_epoch_seconds'] == 10.
    assert result['post_first_epoch_seconds']['count'] == 39
    assert result['post_first_epoch_seconds']['median'] == 30.
    assert result['complete_stage_presentations_per_second'] == 3998720 / 2000
    assert result['evaluation_stage_fraction'] == .04
    assert 'not synchronized GPU compute' in result['optimizer_scope']
    phases['training_batches'] = 31239
    with pytest.raises(ValueError, match='Phase counters'):
        timing_summary(rows, 2000., phases)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1., 0.])
def test_execution_timing_report_rejects_invalid_epoch_timings(value):
    from molgap.gptrans_execution_acceptance import timing_summary
    rows = [{'elapsed_seconds': 10.} for _ in range(40)]
    rows[3]['elapsed_seconds'] = value
    with pytest.raises(ValueError, match='finite timings'):
        timing_summary(rows, 2000., {})


def test_recorded_schedule_allows_only_libm_roundoff_in_local_recomputation():
    import math
    from molgap.gptrans_execution_acceptance import validate_recorded_learning_rate
    from molgap.pcqm_gptrans_v4 import FrozenEpochScheduler
    expected = FrozenEpochScheduler.learning_rate(18)
    adjacent = math.nextafter(expected, math.inf)
    validate_recorded_learning_rate(18, expected, expected)
    validate_recorded_learning_rate(18, adjacent, adjacent)
    with pytest.raises(ValueError, match='learning-rate'):
        validate_recorded_learning_rate(18, adjacent, expected)
    with pytest.raises(ValueError, match='learning-rate'):
        validate_recorded_learning_rate(18, expected + 3 * math.ulp(expected),
                                       expected + 3 * math.ulp(expected))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 0., -1., 0.0008])
def test_recorded_schedule_rejects_nonfinite_or_changed_rates(value):
    from molgap.gptrans_execution_acceptance import validate_recorded_learning_rate
    with pytest.raises(ValueError, match='learning-rate'):
        validate_recorded_learning_rate(18, value, value)
