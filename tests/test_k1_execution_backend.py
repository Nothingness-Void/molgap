"""Small synthetic K1 execution-provider integration; no molecular cache access."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch_geometric.data import Batch, Data

from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_data import DatasetSelection
from molgap.k1_execution_backend import K1ExecutionConfig, execution_context
from molgap.k1_execution_layout import cpu_layout_context
from molgap.qm9_neural_atom import make_encoder


@pytest.fixture(autouse=True)
def runtime():
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.set_rng_state(rng)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)
        torch.utils.deterministic.fill_uninitialized_memory = fill


def batch():
    return Batch.from_data_list([Data(x=torch.zeros(3,9,dtype=torch.long),
        edge_index=torch.tensor([[1,0,1,2,1],[0,1,2,1,2]]),
        edge_attr=torch.zeros(5,3,dtype=torch.long), random_walk_pe=torch.zeros(3,16),
        y=torch.tensor([1.]), source_idx=torch.tensor([i])) for i in range(2)])


def recipe():
    return FamilyRecipe('neural_atom_k1','reference',42,'1'*64,
        DatasetSelection('synthetic.json','2'*64,False),128,0,2,.001,.01,
        ScheduleSpec('cosine_step',2),'final-live','python-epoch-shuffle',1,
        target_mean_eV=0.,target_std_eV=1.,torch_version='2.7.1')


def test_execution_options_are_explicit_identity_and_legacy_bytes_are_stable():
    legacy=recipe()
    assert 'k1_execution' not in legacy.to_dict()
    assert 'k1_execution' not in legacy.execution_policy
    assert FamilyRecipe.from_dict(legacy.to_dict())==legacy
    fast=replace(legacy,k1_execution=K1ExecutionConfig())
    assert fast.identity!=legacy.identity
    assert FamilyRecipe.from_dict(fast.to_dict())==fast
    for field,value in [('aggregation','unknown'),('pooling','unknown'),('projection','unknown'),
            ('evaluation_memory_fill',1)]:
        with pytest.raises((ValueError,TypeError)):
            K1ExecutionConfig.from_dict({**K1ExecutionConfig().to_dict(),field:value})
    with pytest.raises(ValueError):
        K1ExecutionConfig.from_dict({'aggregation':'cached_scatter'})
    with pytest.raises(ValueError):
        replace(legacy,family='gptrans_t',k1_execution=K1ExecutionConfig())


@pytest.mark.parametrize('training',[False,True])
@pytest.mark.parametrize('provider',['cached_scatter','segmented'])
def test_cached_scatter_preserves_original_edges_state_outputs_and_gradients(training,provider):
    torch.manual_seed(43)
    model=make_encoder('neural_atom_k1')
    initial=deepcopy(model.state_dict())
    data=batch();before=deepcopy(data.to_dict())
    rng=torch.get_rng_state().clone()
    outcomes=[]
    forwards=[block.forward.__func__ for block in model.local_blocks]
    for config in (None,K1ExecutionConfig(aggregation=provider,pooling=provider)):
        model.load_state_dict(initial);model.train(training);model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        with cpu_layout_context(model,data,'cpu'),execution_context(model,data,'cpu',config):
            result=model(data.x,data.edge_index,data.edge_attr,data.batch,data.random_walk_pe)
            result.sum().backward()
        outcomes.append((result.detach(),{k:p.grad.clone() if p.grad is not None else None
            for k,p in model.named_parameters()},deepcopy(model.state_dict()),torch.get_rng_state().clone()))
        assert not hasattr(model,'_k1_execution_active')
        assert '_pool' not in vars(model)
        assert all('aggregate' not in vars(block.conv) for block in model.local_blocks)
        assert [block.forward.__func__ for block in model.local_blocks]==forwards
    for key,value in before.items():assert torch.equal(data[key],value)
    # CPU scatter/index_put may use a different sum tree; native CUDA admission is exact.
    torch.testing.assert_close(outcomes[0][0],outcomes[1][0],rtol=2e-6,atol=2e-7)
    for key,value in outcomes[0][1].items():
        other=outcomes[1][1][key]
        if value is None:assert other is None
        else:torch.testing.assert_close(value,other,rtol=2e-5,atol=2e-6)
    assert outcomes[0][2].keys()==outcomes[1][2].keys()
    assert torch.equal(outcomes[0][3],outcomes[1][3])


def test_execution_context_restores_instance_methods_on_exception_and_nested_rejection():
    model=make_encoder('neural_atom_k1');data=batch()
    for nested in (False,True):
        with pytest.raises((ValueError,RuntimeError)):
            with execution_context(model,data,'cpu',K1ExecutionConfig(projection='node')):
                if nested:
                    with execution_context(model,data,'cpu',K1ExecutionConfig()):pass
                raise RuntimeError('injected')
        assert not hasattr(model,'_k1_execution_active')
        assert '_pool' not in vars(model)
        assert all('aggregate' not in vars(block.conv) for block in model.local_blocks)
        assert all(not hasattr(update,'_execution_project_nodes') for update in model.edge_updates)
        assert all('forward' not in vars(update) for update in model.edge_updates)


def test_pooling_binds_the_transferred_batch_instead_of_the_original_cpu_tensor():
    model=make_encoder('neural_atom_k1');data=batch()
    original_membership=data.batch
    hidden=torch.randn(data.num_nodes,192)
    expected=torch.stack([hidden[original_membership==i].mean(0) for i in range(2)])
    with execution_context(model,data,'cpu',K1ExecutionConfig()):
        # PyG Batch.to mutates/returns the same Batch. A clone simulates the
        # changed tensor identity of H2D without requiring a local accelerator.
        data.batch=data.batch.clone()
        assert data.batch is not original_membership
        torch.testing.assert_close(model._pool(hidden,data.batch,size=2),expected)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='native CUDA is unavailable')
def test_kernel_default_execution_providers_survive_actual_cuda_transfer(tmp_path,monkeypatch):
    from molgap.family_training_kernel import FamilyKernel
    from molgap.v4_runtime import state_dict_sha256
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    model=make_encoder('neural_atom_k1')
    state=model.state_dict();path=tmp_path/'initial.pt';torch.save(state,path)
    config=replace(recipe(),initialization_sha256=state_dict_sha256(state),
        k1_execution=K1ExecutionConfig())
    kernel=FamilyKernel(config,path,device='cuda')
    data=batch();membership=data.batch
    try:
        loss,absolute,rows=kernel.step(data,check_finite=True)
        assert rows==2 and torch.isfinite(loss) and torch.isfinite(absolute)
        kernel.model.eval()
        with torch.no_grad():output=kernel.forward(data)
        assert output.shape==(2,) and torch.isfinite(output).all()
        assert data.batch is membership and data.batch.device.type=='cpu'
    finally:kernel.close()


def test_kernel_fast_evaluation_scopes_fill_and_restores_flags(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from molgap.family_training_kernel import FamilyKernel
    from molgap.v4_runtime import state_dict_sha256
    from molgap import training_reproducibility as owner
    model=make_encoder('neural_atom_k1')
    state=model.state_dict();identity=state_dict_sha256(state)
    path=tmp_path/'initial.pt';torch.save(state,path)
    config=replace(recipe(),initialization_sha256=identity,k1_execution=K1ExecutionConfig())
    kernel=FamilyKernel(config,path,device='cpu');kernel.model.eval()
    calls=[];original=owner.deterministic_memory_fill_context
    @contextmanager
    def observed(*,enabled):
        with original(enabled=enabled):
            calls.append(torch.utils.deterministic.fill_uninitialized_memory)
            yield
    monkeypatch.setattr(owner,'deterministic_memory_fill_context',observed)
    before=torch.utils.deterministic.fill_uninitialized_memory
    with torch.no_grad():
        actual=kernel.forward(batch())
    assert actual.shape==(2,) and torch.isfinite(actual).all()
    assert calls==[False]
    assert torch.utils.deterministic.fill_uninitialized_memory==before
    assert not hasattr(kernel.model,'_k1_execution_active')
    assert len(kernel._clip_parameters)==len(tuple(kernel.model.parameters()))
    kernel.close()


def test_projection_provider_preserves_algebra_and_restores_forward_identity():
    from molgap.gps import _PersistentEdgeUpdate
    host=torch.nn.Module()
    host.local_blocks=torch.nn.ModuleList()
    host.edge_updates=torch.nn.ModuleList([_PersistentEdgeUpdate(8,4,.2).double()])
    update=host.edge_updates[0]
    original=update.forward.__func__
    hidden=torch.randn(3,8,dtype=torch.double)
    edges=torch.tensor([[0,1,1,2,1],[1,0,2,1,2]])
    initial=deepcopy(update.state_dict())
    rng=torch.get_rng_state().clone()
    outcomes=[]
    for config in (None,K1ExecutionConfig(aggregation='native',pooling='native',projection='node')):
        update.load_state_dict(initial);update.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        value=hidden.clone().requires_grad_()
        with execution_context(host,None,'cpu',config):
            output=update(value,edges,torch.ones(5,4,dtype=torch.double))
            output.square().sum().backward()
        outcomes.append((output.detach(),value.grad.clone(),
            [p.grad.clone() for p in update.parameters()],torch.get_rng_state().clone()))
        assert update.forward.__func__ is original
    for a,b in zip(outcomes[0][:2],outcomes[1][:2]):
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)
    for a,b in zip(outcomes[0][2],outcomes[1][2]):
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)
    assert torch.equal(outcomes[0][3],outcomes[1][3])
