"""Bind reviewed worker bytes, verify frozen inputs, execute bounded CPU work."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import tarfile
import hashlib
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p,v):
    from molgap.training_reproducibility import atomic_json
    atomic_json(p,v)

def verify_inputs():
    v=read(HERE/'inputs.json')
    for key in ('checkpoints','predictions'):
        for item in v[key].values():
            if sha(item['path'])!=item['sha256']: raise ValueError('Retained tensor changed')
    c=v['cache']
    if sha(Path(c['root'])/'manifest.json')!=c['manifest_sha256']: raise ValueError('Manifest changed')
    if sha(c['development_shard']['path'])!=c['development_shard']['sha256']: raise ValueError('Shard changed')
    s=v['source']; source=Path(s['pythonpath'])
    if sha(s['archive'])!=s['archive_sha256']: raise ValueError('Archive changed')
    with tarfile.open(s['archive']) as archive:
        entries={m.name:m for m in archive.getmembers() if m.isfile()}
        for relative,digest in s['files'].items():
            if sha(source/relative)!=digest: raise ValueError('Frozen module changed')
            member=entries.get('src/'+relative)
            if member is None or hashlib.sha256(archive.extractfile(member).read()).hexdigest()!=digest:
                raise ValueError('Extracted source differs from frozen archive')
    return v

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['freeze','run'])
    args=parser.parse_args()
    snapshot=HERE/'frozen_execution.json'
    names=['measure.py','structure.py','launch.py','inputs.json','protocol.md','roles.json','rml/trajectory.json']
    if args.mode=='freeze':
        if snapshot.exists():raise FileExistsError('Execution snapshot already frozen')
        verify_inputs()
        write(snapshot,{'format':'molgap-k1-slot-execution-freeze-v1','source_hashes':{n:sha(HERE/n) for n in names},
            'parent_review':'Exact loader, strict states, observational hooks, development-only graph access reviewed',
            'user_authority':'2026-10-02 做分析吧','training_released':False})
        print('FROZEN_NOT_EXECUTED')
        return
    frozen=read(snapshot)
    for name,digest in frozen['source_hashes'].items():
        if sha(HERE/name)!=digest:raise ValueError('Reviewed worker/contract changed')
    inputs=verify_inputs()
    env=dict(os.environ,PYTHONPATH=inputs['source']['pythonpath'])
    reference=Path(inputs['checkpoints']['width192']['path']).parent
    candidate=Path(inputs['checkpoints']['width256']['path']).parent
    commands=[['measure.py','--source-root',inputs['source']['pythonpath'],'--cache',inputs['cache']['root'],
        '--reference',str(reference),'--candidate',str(candidate),'--output',str(HERE/'results/measurement')],
        ['structure.py','--trajectory',str(HERE/'rml/trajectory.json'),'--trajectory-sha256',sha(HERE/'rml/trajectory.json'),
         '--output',str(HERE/'results/structure')]]
    wall=time.perf_counter(); observed=[]
    for command in commands:
        result=subprocess.run([sys.executable,str(HERE/command[0]),*command[1:]],env=env,cwd=ROOT,timeout=660)
        observed.append({'worker':command[0],'exit_code':result.returncode})
        write(HERE/'results/execution.json',{'workers':observed,'status':'complete' if len(observed)==2 and
            all(o['exit_code']==0 for o in observed) else 'pending_or_failed',
            'wall_seconds':time.perf_counter()-wall,'frozen_execution_sha256':sha(snapshot)})
        if result.returncode:raise RuntimeError('Diagnostic failed; no automatic retry')
    print('COMPLETE_NO_TRAIN')

if __name__=='__main__':main()
