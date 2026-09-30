"""Small stage bridge; existing hierarchy and family trainers own their loops."""
from pathlib import Path
import copy
import time
import torch
from .training_reproducibility import capture_rng_state, restore_rng_state, atomic_json, sha256_file
from .local_hierarchy_cache import LabelledTrainingGraphs
from .qm9_local_hierarchy import pretrain_local_hierarchy
from .local_hierarchy_adapter import LocalHierarchyAdapter, mask_batch, HierarchyConfig
from .v4_runtime import model_state_sha256, certify_numerical_repeatability


def qualify_stage(model, graphs, *, family, output):
    """Small GPU-only check for the new feature/head path; preserve initial state."""
    from torch_geometric.loader import DataLoader
    if not torch.cuda.is_available():
        raise RuntimeError('Hierarchy runtime qualification requires CUDA')
    output.mkdir(parents=True, exist_ok=True)
    rng=capture_rng_state()
    state=copy.deepcopy(model.state_dict())
    adapter=LocalHierarchyAdapter(model, family)
    head_state=copy.deepcopy(adapter.heads.state_dict())
    batch=next(iter(DataLoader(graphs,batch_size=128,shuffle=False)))
    masks=torch.Generator().manual_seed(59)
    masked,nodes,edges=mask_batch(batch,masks,HierarchyConfig())
    original=batch.to('cuda'); masked=masked.to('cuda')
    nodes=nodes.to('cuda'); edges=edges.to('cuda')
    repeat_rng=capture_rng_state()
    losses,states=[],[]
    torch.cuda.reset_peak_memory_stats()
    tick=time.perf_counter()
    try:
        for _ in range(2):
            model.load_state_dict(state,strict=True)
            adapter.heads.load_state_dict(head_state,strict=True)
            restore_rng_state(repeat_rng)
            model.train()
            parameters=list(model.parameters())+list(adapter.heads.parameters())
            optimizer=torch.optim.AdamW(parameters,lr=1.6e-4,weight_decay=1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss,_=adapter.loss(masked,original,nodes,edges)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            states.append({**{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                           **{'hierarchy.'+k:v.detach().cpu().clone() for k,v in adapter.heads.state_dict().items()}})
        repeat=certify_numerical_repeatability(losses=losses,states=states,maximum_loss_delta=1e-7,maximum_parameter_delta=1e-7)
        peak=torch.cuda.max_memory_allocated(); total=torch.cuda.get_device_properties(0).total_memory
        if peak/total>.85: raise RuntimeError('Hierarchy BS128 memory reserve below 15%')
        record={'accepted':True,'family':family,'repeatability':repeat,'peak_bytes':peak,
                'total_bytes':total,'wall_seconds':time.perf_counter()-tick,
                'scope':'training-role-first-batch-hierarchy-path-only'}
        atomic_json(output/'hierarchy_preflight.json',record)
        return record
    finally:
        model.load_state_dict(state,strict=True)
        model.zero_grad(set_to_none=True)
        restore_rng_state(rng)


def apply_pretraining(model, graphs, *, family, output, config, source_commit):
    if set(config) != {'label_root','label_manifest_sha256','epochs','initialization_sha256'} or config['epochs'] != 10:
        raise ValueError('Unsupported frozen pretraining configuration')
    if model_state_sha256(model)!=config['initialization_sha256']:
        raise ValueError('Frozen family initialization changed before pretraining')
    labelled=LabelledTrainingGraphs(graphs,Path(config['label_root']),config['label_manifest_sha256'])
    rng=capture_rng_state()
    head=model.head if family=='k1' else model.readout
    head_state={k:v.detach().clone() for k,v in head.state_dict().items()}
    try:
        qualify_stage(model,labelled,family=family,output=output)
        model,record=pretrain_local_hierarchy(model,{'train':labelled},output,
            seed=42,source_commit=source_commit,cache_sha256=config['label_manifest_sha256'],
            family=family,epochs=10,batch_size=128,drop_last=True,
            learning_rate=1.6e-4,weight_decay=1e-6)
    finally:
        restore_rng_state(rng)
    head.load_state_dict(head_state,strict=True)
    atomic_json(output/'stage_complete.json',{**record,'family':family,
        'label_manifest_sha256':config['label_manifest_sha256'],
        'source_commit':source_commit,'downstream_optimizer_reset':True,
        'downstream_rng_restored':True,'gap_head_restored':True})
    return model
