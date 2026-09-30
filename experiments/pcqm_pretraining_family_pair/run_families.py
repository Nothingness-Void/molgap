"""Thin two-device dispatcher into existing hierarchy and family trainers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import time


def one(name):
    paths=list(Path('/kaggle/input').rglob(name))
    if len(paths)!=1: raise RuntimeError(f'Expected one {name}: {len(paths)}')
    return paths[0]


def main():
    source=Path(os.environ['MOLGAP_SOURCE_ROOT'])
    sys.path.insert(0,str(source/'src'))
    family=os.environ.get('MOLGAP_HIERARCHY_ARM')
    if family is None:
        from molgap.training_reproducibility import atomic_json
        subprocess.check_call([sys.executable,'-m','pip','install','-q','--no-deps','torch-geometric==2.6.1','ogb==1.3.6'])
        import torch
        if torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
            raise RuntimeError('Contract requires two T4 devices')
        atomic_json(Path('/kaggle/working/startup.json'),{'gpu_names':[torch.cuda.get_device_name(i) for i in range(2)],'family_trainers':['pcqm_k1_variants_runner.train_arm','noisy_nodes.run_training_noisy_nodes']})
        workers=[]
        for device,arm in enumerate(['k1','gptrans_joint']):
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(device),'MOLGAP_HIERARCHY_ARM':arm,'PYTHONPATH':str(source/'src')}
            workers.append(subprocess.Popen([sys.executable,__file__],env=env))
        results=[p.wait() for p in workers]
        atomic_json(Path('/kaggle/working/worker_exits.json'),dict(zip(['k1','gptrans_joint'],results)))
        if any(results): raise RuntimeError(f'Independent arm exit codes: {results}')
        return
    from molgap.training_reproducibility import sha256_file
    from molgap.experiment_spec import ExperimentSpec
    release=json.loads(one('hierarchy_release.json').read_text())
    spec=ExperimentSpec.from_json(one('experiment_spec.json').read_text())
    if spec.identity!=release['spec_identity']: raise RuntimeError('Spec identity changed')
    labels=one('hierarchy_manifest.json').parent
    initial=one('initial_state.pt')
    fixed=[p for p in Path('/kaggle/input').rglob('manifest.json') if sha256_file(p)==release['fixed_manifest_sha256']]
    if len(fixed)!=1: raise RuntimeError('Missing unique fixed V4 dataset')
    config={'label_root':str(labels),'label_manifest_sha256':release['label_manifest_sha256'],'epochs':10,
            'initialization_sha256':release['initializations'][family]}
    output=Path('/kaggle/working')/family
    if family=='k1':
        os.environ['MOLGAP_FIXED_CACHE_ROOT']=str(fixed[0].parent)
        os.environ['MOLGAP_FIXED_DATASET']='nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1'
        os.environ['MOLGAP_PLATFORM_ID']='kaggle1'
        from molgap.pcqm_k1_variants_runner import train_arm
        train_arm('neural_atom_k1_v4',output,source_commit=release['source_commit'],source_archive_sha256=release['source_archive_sha256'],pretraining=config)
    elif family=='gptrans_joint':
        from molgap.pcqm_gptrans_v4 import run_preflight
        preflight_dir=output/'downstream_preflight'
        run_preflight(dataset_root=fixed[0].parent,manifest_path=fixed[0],
            source_archive=one('source_payload.bin'),source_archive_sha256=release['source_archive_sha256'],
            source_commit=release['source_commit'],output=preflight_dir,
            platform_id='kaggle1-t4',initial_state_path=initial,variant='pair_update_norm')
        from molgap.noisy_nodes import run_training_noisy_nodes
        run_training_noisy_nodes(dataset_root=fixed[0].parent,manifest_path=fixed[0],output=output,
            initial_state_path=initial,source_commit=release['source_commit'],source_archive_sha256=release['source_archive_sha256'],
            preflight_path=preflight_dir/'preflight.json',
            pair_update_norm=True,pretraining=config)
    else: raise ValueError('Unknown arm')


if __name__=='__main__':
    arm=os.environ.get('MOLGAP_HIERARCHY_ARM')
    started=time.time()
    def stop_for_cost(_sig,_frame):
        raise TimeoutError('Frozen 8 T4-hour arm cap; retain latest durable epoch checkpoint')
    if arm:
        signal.signal(signal.SIGALRM,stop_for_cost)
        signal.alarm(8*3600)
    try:
        main()
    finally:
        if arm:
            source=Path(os.environ['MOLGAP_SOURCE_ROOT']);sys.path.insert(0,str(source/'src'))
            from molgap.training_reproducibility import atomic_json
            atomic_json(Path('/kaggle/working')/arm/'native_cost_observation.json',{
                'hardware':'Nvidia Tesla T4','assigned_device':os.environ['CUDA_VISIBLE_DEVICES'],
                'started_unix':started,'finished_unix':time.time(),
                't4_process_wall_seconds':time.time()-started,
                'account':'nothingnessvoid','protected_roles_read':False})
