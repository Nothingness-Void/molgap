"""Thin Colab adapter over frozen payload and shared K1 profile."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    binding=json.loads((HERE/"upload_binding.json").read_text())
    archive_name=Path(binding['file']).name
    attempt=binding.get('attempt_id')
    run_path='V5/runs/k1-profile-a100-20261007'+('/'+attempt if attempt else '')
    def code(text):
        return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":text.splitlines(True)}
    cells=[{"cell_type":"markdown","metadata":{},"source":[
        "# MolGap K1 A100 execution profile\n",
        "Desktop-owned execution diagnostics on training members only. FP32/noTF32.\n",
        "Source, selected checkpoint and sample are hash-bound; no scientific training or model selection.\n",
        "Drive output: `MolGap/V5/runs/k1-profile-a100-20261007/`. Worker ceiling20min.\n"]},
    code('''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import hashlib, json, os, subprocess, sys, time, zipfile
DRIVE_ROOT=Path('/content/drive/MyDrive/MolGap')
RUN_ROOT=DRIVE_ROOT/'RUN_PATH_PLACEHOLDER'
RUN_ROOT.mkdir(parents=True,exist_ok=True)
print('Durable output:',RUN_ROOT,flush=True)
'''.replace('RUN_PATH_PLACEHOLDER',run_path)),code(f'''# Immutable source/input transfer: no credentials or dataset download.
ARCHIVE=DRIVE_ROOT/'{archive_name}'
EXPECTED_SHA256='{binding['sha256']}'
assert ARCHIVE.is_file(), 'Upload the prepared ZIP to MyDrive/MolGap first.'
assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==EXPECTED_SHA256
PAYLOAD=Path('/content/k1-profile-a100-20261007')
PAYLOAD.mkdir(exist_ok=True)
with zipfile.ZipFile(ARCHIVE) as z:
    for name in z.namelist():
        target=(PAYLOAD/name).resolve()
        assert target.is_relative_to(PAYLOAD.resolve()), 'Unsafe ZIP path'
    z.extractall(PAYLOAD)
manifest=json.loads((PAYLOAD/'payload_manifest.json').read_text())
for name,digest in manifest['files'].items():
    assert hashlib.sha256((PAYLOAD/name).read_bytes()).hexdigest()==digest, name
import shutil
shutil.copy2(PAYLOAD/'payload_manifest.json',RUN_ROOT/'payload_manifest.json')
shutil.copy2(PAYLOAD/'protocol.md',RUN_ROOT/'protocol.md')
print('PAYLOAD_IDENTITY_VERIFIED',len(manifest['sample_source_idx']),flush=True)
'''),code('''# Isolated retained Python3.11/Torch2.4.1 dependencies; notebook kernel stays intact.
setup_started=time.time()
print('Preparing isolated retained runtime',flush=True)
subprocess.run([sys.executable,'-m','pip','install','uv==0.10.9'],check=True,timeout=180)
ENV=Path('/content/k1-profile-python311')
if not (ENV/'bin/python').exists():
    subprocess.run(['uv','venv','--python','3.11',str(ENV)],check=True,timeout=180)
PYTHON=str(ENV/'bin/python')
subprocess.run(['uv','pip','install','--python',PYTHON,'torch==2.4.1',
    '--index-url','https://download.pytorch.org/whl/cu121'],check=True,timeout=480)
subprocess.run(['uv','pip','install','--python',PYTHON,'numpy==1.26.4',
    'torch-geometric==2.6.1','ogb==1.3.6'],check=True,timeout=300)
(RUN_ROOT/'setup_observation.json').write_text(json.dumps({'setup_wall_seconds':time.time()-setup_started,
    'python_environment':PYTHON,'setup_allocated_device_hours_unknown':True},sort_keys=True))
print('ISOLATED_RUNTIME_READY',flush=True)
'''),code('''# One bounded worker. Existing result/runtime requires reconciliation, never blind rerun.
assert not (RUN_ROOT/'runtime.json').exists(), 'Reconcile existing attempt before rerun.'
env=os.environ.copy()
env['PYTHONPATH']=str(PAYLOAD/'src')
env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
env['PYTHONHASHSEED']='42'
log=RUN_ROOT/'worker.log'
from threading import Timer
command=[PYTHON,'-u','-m','molgap.k1_execution_profile','--root',str(PAYLOAD),'--output',str(RUN_ROOT)]
with log.open('w') as handle:
    process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    watchdog=Timer(1200,process.kill)
    watchdog.start()
    worker_started=time.time()
    try:
        for line in process.stdout:
            handle.write(line); handle.flush(); print(line,end='',flush=True)
        returncode=process.wait()
    finally:
        watchdog.cancel()
        (RUN_ROOT/'worker_process_observation.json').write_text(json.dumps({
            'worker_wall_seconds_including_imports':time.time()-worker_started,
            'returncode':process.returncode,'wall_ceiling_seconds':1200},sort_keys=True))
assert returncode==0, f'Worker failed({returncode}); durable evidence: {RUN_ROOT}'
print('DURABLE_K1_PROFILE_COMPLETE',flush=True)
'''),code('''# Verify all bounded results before interpreting them.
completion=json.loads((RUN_ROOT/'completion.json').read_text())
for name,digest in completion['artifacts'].items():
    assert hashlib.sha256((RUN_ROOT/name).read_bytes()).hexdigest()==digest, name
result=json.loads((RUN_ROOT/'result.json').read_text())
for case in result['cases']:
    print(case['case'],'median step seconds:',round(case['median_step_s'],6))
print('Gradient relation:',result['gradient_relation'])
print('Actual worker cost:',result['cost'])
print('Saved in Google Drive:',RUN_ROOT)
''')]
    notebook={"cells":cells,"nbformat":4,"nbformat_minor":5,
        "metadata":{"accelerator":"GPU","colab":{"name":"MolGap_K1_A100_Profile.ipynb","gpuType":"A100"},
            "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
            "language_info":{"name":"python"}}}
    (HERE/'MolGap_K1_A100_Profile.ipynb').write_text(json.dumps(notebook,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
