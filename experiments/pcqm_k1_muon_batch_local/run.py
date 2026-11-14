"""Bounded optimizer and shared-K1-kernel diagnostic, with synthetic inputs only."""
from __future__ import annotations

from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = Path('D:/molgap-artifacts/local_profiles/k1-muon-batch-20261106-attempt002')
WARMUP, STEPS, LIMIT = 4, 24, 600


def main():
    from molgap.research_memory.schemas import validate_trajectory
    from molgap.research_memory.trace import atomic_write, file_digest, json_bytes

    def save(path, value):
        atomic_write(path, json_bytes(value))

    OUT.mkdir(parents=True, exist_ok=False)
    template = json.loads((ROOT / 'experiments/pcqm_k1_hop_fusion_qualification/plan_input.json').read_text())['trajectory']
    prior_path = ROOT / 'experiments/pcqm_k1_muon_100k_20261106/runs/pcqm-k1-muon-100k-20261106-s42/prospective/muon/rml_finalized/trajectory.json'
    prior = json.loads(prior_path.read_text())
    rel = HERE.relative_to(ROOT).as_posix()
    source = {p.relative_to(ROOT).as_posix(): file_digest(p) for p in
              (Path(__file__), ROOT / 'src/molgap/k1_muon.py', HERE / 'protocol.md')}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    template.update(trajectory_id='TB-k1-muon-batch-local-20261106', family_id='k1-muon-batched-execution',
                    question='Does same-shape FP32 batching reduce Muon execution cost with small measured rounding error?')
    template['hypothesis'].update(hypothesis_id='H-k1-muon-batch-local',
        observed_deficiency='The accepted100K profile measures optimizer host spans of228.9ms versus8.4ms for AdamW.',
        alternative_explanations=['Batched reductions alter optimizer numerics materially', 'Stacking and copies outweigh launch savings'],
        changed_mechanism='Same-shaped matrices share FP32 batched operations; optimizer mathematics unchanged.',
        cheapest_falsifier='Seeded gradients on actual K1 shapes and bounded synthetic FamilyKernel windows.',
        supporting_evidence_ids=prior['result']['evidence_ids'], related_closed_family_ids=[prior['family_id']],
        expected_native_cost_ref=f'{rel}/protocol.md',
        decision_changed_if_positive='Retain execution candidate for separate quality qualification; no remote mutation.',
        decision_changed_if_negative='Keep current optimizer without another full training experiment.')
    template['state_at_start'].update(source_commit=commit, source_config_identity=file_digest(Path(__file__)),
        contract_refs=[f'{rel}/protocol.md'], reference_ids=prior['result']['evidence_ids'],
        prior_evidence_ids=prior['result']['evidence_ids'], parent_trajectory_ids=[prior['trajectory_id']],
        role_snapshot_refs=[f'{rel}/protocol.md'], budget_snapshot_ref=f'{rel}/protocol.md')
    template['actions']=[dict(action_id='A001', type='local_execution_profile', run_ids=['local-k1-muon-batch'],
        attempt_ids=['v2'], source_commit=commit, evidence_refs=[f'{rel}/protocol.md'], cost_event_ids=[])]
    template['result']={'evidence_ids': [], 'evidence_refs': []}
    template['decision']=dict(outcome='ACTIVE', decision_ref=f'{rel}/protocol.md',
        next_allowed_actions=['Synthetic local diagnostic only; maximum600process-wall seconds'], reopen_conditions=[])
    template.update(comparison_class='CONTEXT_ONLY', comparison_readiness_ref=f'{rel}/protocol.md',
        comparison_blockers=['Execution-only; synthetic inputs; no molecular accuracy or native T4 qualification'], reference_bundle_id=None)
    template['diagnostic_source_sha256']=source
    validate_trajectory(template)
    save(HERE / 'trajectory.json', template)
    with zipfile.ZipFile(OUT / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in source:
            archive.write(ROOT / name, name)

    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from torch_geometric.data import Batch
    from molgap.family_execution_diagnostic import synthetic_graphs
    from molgap.family_training_data import DatasetSelection
    from molgap.family_training_kernel import FamilyKernel, ContractSchedule
    from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
    from molgap.k1_muon import MuonAdamW, BatchedMuonAdamW, parameter_groups
    from molgap.qm9_neural_atom import make_encoder
    from molgap.training_reproducibility import configure_fp32_determinism, atomic_torch_save
    from molgap.v4_runtime import state_dict_sha256

    torch.set_num_threads(2)
    configure_fp32_determinism(42)
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 5060'
    started=time.perf_counter()
    report=dict(status='RUNNING', hardware=torch.cuda.get_device_name(0), torch=str(torch.__version__),
        cuda=torch.version.cuda, fp32=True, tf32=False, molecular_roles_consumed=False,
        scientific_training=False, source_sha256=source, windows=[], numerical={})
    save(OUT/'result.json', report)

    def deadline():
        if time.perf_counter()-started > LIMIT:
            raise TimeoutError('Local600s diagnostic limit reached')

    def differences(left, right):
        exact=True
        max_abs=0.
        squares=0.
        count=0
        def visit(a,b):
            nonlocal exact,max_abs,squares,count
            if isinstance(a,torch.Tensor):
                assert a.shape==b.shape and a.dtype==b.dtype
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                exact=exact and torch.equal(a,b)
                delta=(a.detach().double()-b.detach().double()).abs()
                max_abs=max(max_abs,float(delta.max()) if delta.numel() else 0.)
                squares+=float(delta.square().sum())
                count+=delta.numel()
            elif isinstance(a,dict):
                assert a.keys()==b.keys()
                for key in a: visit(a[key],b[key])
            elif isinstance(a,(list,tuple)):
                assert len(a)==len(b)
                for x,y in zip(a,b): visit(x,y)
            else:
                assert a==b,(a,b)
        visit(left,right)
        return dict(exact=exact,max_abs=max_abs,rms=(squares/max(count,1))**.5,elements=count)

    configure_fp32_determinism(42)
    initial_model=make_encoder('neural_atom_k1')
    initial=deepcopy(initial_model.state_dict())
    initial_path=OUT/'initial.pt'
    atomic_torch_save(initial_path, initial)
    group_shapes={}
    for p in parameter_groups(initial_model,1e-5)[0]['params']:
        key=str(tuple(p.shape))
        group_shapes[key]=group_shapes.get(key,0)+1
    report['muon_groups']=group_shapes
    report['muon_matrices']=sum(group_shapes.values())
    save(OUT/'result.json',report)
    del initial_model
    classes={'reference':MuonAdamW,'batched':BatchedMuonAdamW}

    def optimizer_trial(kind, timed):
        configure_fp32_determinism(42)
        model=make_encoder('neural_atom_k1').cuda()
        model.load_state_dict(initial)
        optimizer=classes[kind](parameter_groups(model,1e-5),lr=4e-4,weight_decay=1e-5)
        for p in model.parameters(): p.grad=torch.randn_like(p)*.01
        gradients=[p.grad.clone() for p in model.parameters()]
        durations=[]
        for step in range(WARMUP+STEPS if timed else 20):
            deadline()
            torch.cuda.synchronize()
            if timed and step==WARMUP: torch.cuda.reset_peak_memory_stats()
            stamp=time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-stamp
            if timed and step>=WARMUP: durations.append(elapsed)
        memory=torch.cuda.max_memory_allocated()
        state={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
               'optimizer':deepcopy(optimizer.state_dict())}
        import io
        stream=io.BytesIO()
        torch.save(state['optimizer'],stream)
        stream.seek(0)
        state['optimizer']=torch.load(stream,map_location='cpu',weights_only=True)
        assert all(torch.equal(p.grad,g) for p,g in zip(model.parameters(),gradients))
        del model,optimizer,gradients
        gc.collect()
        return state,durations,memory

    refs={}
    for kind in ('reference','batched'):
        refs[kind]=optimizer_trial(kind,False)[0]
    report['numerical']['fixed_gradient_20_updates']=differences(refs['reference'],refs['batched'])
    save(OUT/'result.json',report)
    del refs

    batch=Batch.from_data_list(synthetic_graphs(dict(rows=128,minimum_nodes=8,maximum_nodes=32)))
    recipe=FamilyRecipe('neural_atom_k1','reference',42,state_dict_sha256(initial),
        DatasetSelection('synthetic-only.json','2'*64,False),128,0,WARMUP+STEPS,
        4e-4,1e-5,ScheduleSpec('cosine_step',WARMUP+STEPS),'final-live','python-epoch-shuffle',
        1000,0.,1.,ema_decay=.999,torch_version=str(torch.__version__).split('+')[0],
        optimizer='k1-muon-adamw-fp32-v1')

    def kernel_trial(kind):
        configure_fp32_determinism(42)
        kernel=FamilyKernel(recipe,initial_path,device='cuda:0')
        if kind=='batched':
            old=kernel.optimizer.state_dict()
            kernel.optimizer=BatchedMuonAdamW(parameter_groups(kernel.model,1e-5),lr=4e-4,weight_decay=1e-5)
            kernel.optimizer.load_state_dict(old)
        schedule=ContractSchedule(kernel)
        kernel.model.train()
        losses=[]
        durations=[]
        for step in range(WARMUP+STEPS):
            deadline()
            torch.cuda.synchronize()
            if step==WARMUP: torch.cuda.reset_peak_memory_stats()
            stamp=time.perf_counter()
            loss,_,_=kernel.step(batch,check_finite=False)
            schedule.after_step(step+1)
            torch.cuda.synchronize()
            if step>=WARMUP: durations.append(time.perf_counter()-stamp)
            losses.append(float(loss))
        memory=torch.cuda.max_memory_allocated()
        kernel.model.eval()
        with torch.no_grad(): pred=kernel.forward(batch).detach().cpu()
        endpoint={'losses':torch.tensor(losses),'prediction':pred,
                  'model':{k:v.detach().cpu() for k,v in kernel.model.state_dict().items()}}
        kernel.close()
        del kernel
        gc.collect()
        return endpoint,durations,memory

    endpoints={}
    for mode in ('optimizer','kernel'):
        for index,kind in enumerate(('reference','batched','batched','reference')):
            state,durations,memory=(optimizer_trial(kind,True) if mode=='optimizer' else kernel_trial(kind))
            if mode=='kernel' and kind not in endpoints: endpoints[kind]=state
            row=dict(mode=mode,implementation=kind,order=index,warmup=WARMUP,steps=STEPS,
                     median_seconds=statistics.median(durations),mean_seconds=statistics.mean(durations),
                     raw_seconds=durations,peak_allocated_bytes=memory)
            report['windows'].append(row)
            save(OUT/'result.json',report)
            print(json.dumps({k:v for k,v in row.items() if k!='raw_seconds'}),flush=True)
            del state
    for key in ('losses','prediction','model'):
        report['numerical']['shared_kernel_'+key]=differences(endpoints['reference'][key],endpoints['batched'][key])
    report.update(status='COMPLETE',elapsed_seconds=time.perf_counter()-started)
    save(OUT/'result.json',report)
    save(HERE/'result.json',report)
    print(json.dumps(report['numerical']),flush=True)


if __name__=='__main__':
    main()
