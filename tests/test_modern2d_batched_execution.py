"""Synthetic CPU execution equivalence, never a scientific score qualification."""
from copy import deepcopy
import pytest
import torch
from torch_geometric.data import Batch
from modern2d_fixtures import graph
from molgap.modern2d_inputs import validated_topology_views
from molgap.esa_native import ESANative, native_tokens, packed_native_tokens
from test_esa_native import small_config as esa_config, mixed_graph, no_edges
from molgap.ppgt_author import PPGTAuthor
from test_ppgt_author_fidelity import small_config as ppgt_config, fixed_weights


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def diverse():
    a = mixed_graph(9)
    b = graph(3, 10)
    c = mixed_graph(11)
    c.edge_index = c.edge_index.flip(1)
    c.edge_attr = c.edge_attr.flip(0)
    return Batch.from_data_list([a, no_edges(1, 13), b, c, no_edges(3, 15)])


def test_packed_views_preserve_order_duplicates_and_source_identity():
    batch = diverse()
    # Deliberately interleave graph edges; stable grouping cannot reorder within a graph.
    permutation = torch.arange(batch.edge_index.shape[1]).flip(0)
    batch.edge_index, batch.edge_attr = batch.edge_index[:, permutation], batch.edge_attr[permutation]
    original = batch.clone()
    slow = validated_topology_views(batch)
    fast = validated_topology_views(batch, packed=True)
    for a, b in zip(slow, fast):
        assert a.source_idx == b.source_idx
        for key in ('x', 'edge_index', 'edge_attr'):
            assert torch.equal(getattr(a, key), getattr(b, key))
    for key in ('x', 'edge_index', 'edge_attr', 'ptr', 'source_idx'):
        assert torch.equal(batch[key], original[key])


def test_packed_native_tokens_are_bit_exact_to_original_padding():
    from torch.nn.utils.rnn import pad_sequence
    views = validated_topology_views(diverse(), packed=True)
    tokens = [native_tokens(g) for g in views]
    features, types, endpoints, valid = packed_native_tokens(views)
    assert torch.equal(features, pad_sequence([t.features for t in tokens], batch_first=True))
    assert torch.equal(types, pad_sequence([t.token_types for t in tokens], batch_first=True))
    assert torch.equal(endpoints, pad_sequence([t.endpoints.t() for t in tokens], batch_first=True, padding_value=-1))
    assert torch.equal(valid.sum(1), torch.tensor([len(t.token_types) for t in tokens]))


@pytest.mark.parametrize('family', ['esa', 'ppgt'])
def test_forward_gradients_and_optimizer_match_reference(family):
    torch.manual_seed(42)
    model = ESANative(esa_config()) if family == 'esa' else PPGTAuthor(ppgt_config())
    if family == 'ppgt':
        fixed_weights(model)
    reference = deepcopy(model)
    batch = diverse()
    got = model.forward_bucketed(batch) if family == 'ppgt' else model(batch)
    wanted = reference.forward_reference(batch)
    torch.testing.assert_close(got, wanted, atol=3e-6, rtol=2e-5)
    weights = torch.tensor([.7, -.4, .2, .5, -.1])
    (got * weights).sum().backward()
    (wanted * weights).sum().backward()
    for (name, value), (other, expected) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other and value.grad is not None and expected.grad is not None
        torch.testing.assert_close(value.grad, expected.grad, atol=1e-5, rtol=8e-5, msg=name)
    # Check one real optimizer transition, not only forward closeness.
    torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5).step()
    torch.optim.AdamW(reference.parameters(), lr=3e-4, weight_decay=1e-5).step()
    if family == 'esa':
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], atol=3e-6, rtol=2e-5, msg=name)
    else:
        from molgap.modern2d_execution_proof import compare_tensors
        report = compare_tensors(model.state_dict(), reference.state_dict())
        assert report['finite'] and report['schema_equal']
        # A diagnostic reports strict update drift; it must not relabel it as
        # equivalent merely because forward/gradient tolerances passed.
        assert report['update_equivalent'] is False
        assert 'blocks.0.attention.P.bias' in report['mismatches']


@pytest.mark.parametrize('family', ['esa', 'ppgt'])
def test_companion_order_and_batch_permutation_preserved(family):
    model = ESANative(esa_config()) if family == 'esa' else PPGTAuthor(ppgt_config())
    batch = diverse()
    graphs = batch.to_data_list()
    permutation = [4, 2, 0, 3, 1]
    call = model.forward_bucketed if family == 'ppgt' else model
    expected = call(batch)
    observed = call(Batch.from_data_list([graphs[i] for i in permutation]))
    torch.testing.assert_close(observed, expected[permutation], atol=3e-6, rtol=2e-5)


def test_ppgt_stochastic_configuration_retains_graphwise_rng_policy():
    model = PPGTAuthor(ppgt_config(stochastic_depth=.2, attention_dropout=.1)).train()
    batch = diverse()
    torch.manual_seed(24)
    expected = model.forward_reference(batch)
    torch.manual_seed(24)
    assert torch.equal(model(batch), expected)


@pytest.mark.parametrize('change', ['role', 'geometry', 'cross_graph', 'ptr', 'category', 'bounds'])
def test_packed_path_keeps_fail_closed_validation(change):
    batch = diverse()
    kwargs = {}
    if change == 'role': batch.role_name = ['test_dev'] * batch.num_graphs
    elif change == 'geometry': batch.pos = torch.zeros(batch.num_nodes, 3)
    elif change == 'cross_graph': batch.edge_index[1, 0] = len(batch.x) - 1
    elif change == 'ptr': batch.ptr[1] += 1
    elif change == 'category': batch.x[0, 0] = 1000
    else: kwargs['max_nodes'] = 2
    with pytest.raises(ValueError):
        validated_topology_views(batch, packed=True, **kwargs)


def repair_authority():
    from molgap.modern2d_training_release import CONFIGURED_AUTHORITY_FORMAT, REPAIR_PURPOSE
    arms = [{'arm_id': 'ppgt', 'family': {'name': 'ppgt_small', 'version': '3'}},
            {'arm_id': 'esa', 'family': {'name': 'esa_small', 'version': '4'}}]
    gpu = dict(owner='testowner', accelerator='NvidiaTeslaT4', allocated_devices=2,
               wall_ceiling_seconds=1800, allocated_t4_hours_ceiling=1)
    authority = dict(format=CONFIGURED_AUTHORITY_FORMAT, owner='server', purpose=REPAIR_PURPOSE,
        user_authorized=True, scientific_training_authorized=False, logical_run_id='probe',
        arms=['ppgt','esa'], interfaces=[['ppgt_small','3'],['esa_small','4']], seeds=[42],
        physical_batch_size=128, roles=['train_100k','common_dev'], scale_500k_authorized=False,
        protected_role_access_authorized=False, extra_seed_authorized=False,
        automatic_successor_authorized=False, full_training_authorized=False,
        allocated_t4_hours_ceiling=1, gpu=gpu, run_reference='testowner/probe',
        scratch_optimizer_updates_per_model=8, execution_reference_required=True)
    declaration = dict(arms=arms, logical_run_id='probe', platform=dict(name='kaggle',
        accelerator='NvidiaTeslaT4', device_count=2))
    return authority, declaration


def test_repair_release_is_separate_bounded_and_nontraining():
    from molgap.modern2d_training_release import validate_configured_authority
    authority, declaration = repair_authority()
    assert validate_configured_authority(authority, declaration) == 1800


@pytest.mark.parametrize('changed', ['model', 'optimizer', 'scheduler', 'rng_state', 'optimizer_metadata'])
def test_execution_state_report_does_not_hide_component_drift(changed):
    from molgap.modern2d_execution_proof import compare_states
    state = dict(model={'weight':torch.ones(2)}, optimizer={'state':{0:{'exp_avg':torch.ones(2)}},
        'param_groups':[{'lr':.001}]}, scheduler={'completed_units':2}, cursor={'step':2},
        rng_state={'torch':torch.zeros(4,dtype=torch.uint8)})
    reference = deepcopy(state)
    assert compare_states(state, reference)['exact_full_state']
    if changed == 'model': state['model']['weight'][0] = 2
    elif changed == 'optimizer': state['optimizer']['state'][0]['exp_avg'][0] = 2
    elif changed == 'scheduler': state['scheduler']['completed_units'] = 3
    elif changed == 'rng_state': state['rng_state']['torch'][0] = 1
    else: state['optimizer']['param_groups'][0]['lr'] = .1
    report = compare_states(state, reference)
    assert not report['update_equivalent'] and not report['exact_full_state']


@pytest.mark.parametrize('field,value', [('scientific_training_authorized', True),
    ('scratch_optimizer_updates_per_model', 9), ('execution_reference_required', False),
    ('protected_role_access_authorized', True), ('automatic_successor_authorized', True)])
def test_repair_release_rejects_authority_expansion(field, value):
    from molgap.modern2d_training_release import validate_configured_authority
    authority, declaration = repair_authority()
    authority[field] = value
    with pytest.raises(ValueError):
        validate_configured_authority(authority, declaration)
