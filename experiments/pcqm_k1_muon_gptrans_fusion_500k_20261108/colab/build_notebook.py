"""Build a self-contained Colab preparation/launch notebook; never execute it."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
PRIMARY = Path("D:/文档/molgap")
EXPORT = Path("D:/下载/MolGap-Colab-Full-20261108")
SOURCE_COMMIT = "5f55a7b55c54bd59bbf104053316c8a136bb4ba0"


def main():
    EXPORT.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(["git", "archive", "--format=tar.gz", SOURCE_COMMIT,
        "src/molgap", "pyproject.toml"], cwd=PRIMARY)
    manifest = json.loads((PRIMARY / "platforms/_records/ims/pcqm_fixed_datasets_v1/ogb-train-full.manifest.json").read_text())
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    assert manifest_sha == "7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d"
    cells = []
    def add(kind, name, text):
        cell = dict(cell_type=kind, id=name, metadata={}, source=text.splitlines(True))
        if kind == "code":
            ast.parse(text)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "intro", """# MolGap：加速 K1＋GPTrans 全量 Colab

**当前先运行 CPU 准备模式。正式全量训练包尚未冻结，本笔记本不把旧500K包冒充全量。**

1. 在 CPU 运行时用 `MODE='prepare'` 核对云盘中已接受的3378606行训练缓存，保存全量训练标签统计。
2. 取得两份通过 `prepare-family-training` 冻结的 Colab 全量 prepared 包后，切换 A100，设 `MODE='train'`。
3. 单张 A100 顺序执行两个独立任务，共享同一个训练数据副本。K1 使用批量 Muon；GPTrans 使用 BondLocal＋PairNorm。
4. 原子检查点、曲线、原生证明和训练输出直接保存到 Drive。重连后重用原输出及精确状态。
5. 成功结束或单元报错后，自动释放运行时；仍可使用最后的手动释放单元。

只读取全量 TRAIN 拓扑，不读取几何、官方验证或测试。全量训练没有内部开发集，按冻结合同的 final-live/final-ema 输出；本笔记本不自行选择 epoch、改变学习率或补做官方评测。
""")
    add("code", "settings", """from pathlib import Path
import json, os, subprocess, sys, time, tarfile, base64, shutil, signal
from google.colab import runtime

# prepare 模式请使用 CPU；train 模式请在设置中明确选择 A100。
MODE = 'prepare'
AUTO_RELEASE = True
DRIVE_ROOT = Path('/content/drive/MyDrive/MolGap/V5')
FULL_CACHE_ON_DRIVE = DRIVE_ROOT / 'full_data/pcqm4mv2-fixed-v1'
RELEASE_ROOT = DRIVE_ROOT / 'releases/k1-muon-gptrans-full-20261108'
RUN_ROOT = DRIVE_ROOT / 'runs/k1-muon-gptrans-full-20261108'
ARM_ORDER = ('k1_muon', 'gptrans_pairnorm')
SESSION_SECONDS = 10 * 3600
CLEANUP_SECONDS = 300
SESSION_STARTED = time.perf_counter()
active_allocation = None
active_allocation_root = None

def release_after_error(result):
    if AUTO_RELEASE and (result.error_before_exec is not None or result.error_in_exec is not None):
        if active_allocation is not None:
            active_allocation.write(active_allocation_root, 'failed')
        runtime.unassign()

get_ipython().events.register('post_run_cell', release_after_error)
assert MODE in ('prepare', 'train')
print('Mode:', MODE, '| No training is performed in prepare mode.')
""")
    add("code", "mount", """from google.colab import drive
drive.mount('/content/drive')
RUN_ROOT.mkdir(parents=True, exist_ok=True)
if MODE == 'train':
    observed_gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip().splitlines()
    assert len(observed_gpu) == 1 and 'A100' in observed_gpu[0], observed_gpu
    for arm in ARM_ORDER:
        assert (RELEASE_ROOT / arm / 'prepared_training.json').is_file(), f'Full prepared bundle required: {RELEASE_ROOT / arm}'
print('Outputs:', RUN_ROOT)
""")
    add("code", "frozen-source", f"""# 自包含源码快照；不访问移动分支、不包含模型或凭证。
SOURCE_COMMIT = '{SOURCE_COMMIT}'
SOURCE_ARCHIVE_SHA256 = '{hashlib.sha256(source).hexdigest()}'
SOURCE_ARCHIVE_B64 = '{base64.b64encode(source).decode()}'
ACCEPTED_MANIFEST_B64 = '{base64.b64encode(manifest_bytes).decode()}'
ACCEPTED_MANIFEST_SHA256 = '{manifest_sha}'
import hashlib, io
raw_source = base64.b64decode(SOURCE_ARCHIVE_B64)
assert hashlib.sha256(raw_source).hexdigest() == SOURCE_ARCHIVE_SHA256
SOURCE_ROOT = Path('/content/molgap-colab-source-' + SOURCE_COMMIT[:8])
SOURCE_ROOT.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(raw_source), mode='r:gz') as bundle:
    bundle.extractall(SOURCE_ROOT, filter='data')
sys.path.insert(0, str(SOURCE_ROOT / 'src'))
manifest_bytes = base64.b64decode(ACCEPTED_MANIFEST_B64)
assert hashlib.sha256(manifest_bytes).hexdigest() == ACCEPTED_MANIFEST_SHA256
accepted_manifest = json.loads(manifest_bytes)
assert accepted_manifest['roles']['train']['rows'] == 3378606
assert accepted_manifest['roles']['development'] is None
""")
    add("code", "environment", """# 独立环境，避免改变 Colab 内核环境；准备模式安装 CPU Torch。
ENV_ROOT = Path('/content/molgap-full-' + MODE)
subprocess.run([sys.executable, '-m', 'venv', str(ENV_ROOT)], check=True)
PYTHON = str(ENV_ROOT / 'bin/python')
wheel_channel = 'cpu' if MODE == 'prepare' else 'cu128'
subprocess.run([PYTHON, '-m', 'pip', 'install', '--no-cache-dir', 'torch==2.11.0',
    '--index-url', 'https://download.pytorch.org/whl/' + wheel_channel], check=True)
subprocess.run([PYTHON, '-m', 'pip', 'install', '--no-cache-dir',
    'numpy<2', 'torch-geometric==2.6.1', 'ogb==1.3.6'], check=True)
env = dict(os.environ, PYTHONPATH=str(SOURCE_ROOT / 'src'), PYTHONUNBUFFERED='1',
    PYTHONHASHSEED='42', CUBLAS_WORKSPACE_CONFIG=':4096:8',
    OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
from molgap.training_reproducibility import sha256_file, atomic_json
print('Worker Python:', PYTHON)
""")
    add("code", "data-stage", """# 仅复制清单指定的68个TRAIN拓扑分片；既有且hash正确的SSD文件复用。
topology = accepted_manifest['assets']['topology']
assert len(topology) == 68 and all(row['role'] == 'train' for row in topology)
assert sum(row['rows'] for row in topology) == 3378606
DATA_ROOT = Path('/content/molgap-full-data')
DATA_ROOT.mkdir(exist_ok=True)
required = sum(row['bytes'] for row in topology if not (DATA_ROOT / row['file']).is_file())
assert shutil.disk_usage('/content').free > required + 5 * 1024**3, 'Insufficient local SSD space'
for number, row in enumerate(topology, 1):
    destination = DATA_ROOT / row['file']
    if destination.is_file() and sha256_file(destination) == row['sha256']:
        continue
    original = FULL_CACHE_ON_DRIVE / row['file']
    assert original.is_file(), f'Missing accepted full TRAIN shard: {original}'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.staging')
    shutil.copyfile(original, temporary)
    assert sha256_file(temporary) == row['sha256'], original
    os.replace(temporary, destination)
    print('TRAIN shard', number, '/', len(topology), flush=True)
(DATA_ROOT / 'manifest.json').write_bytes(manifest_bytes)
atomic_json(RUN_ROOT / 'cache_staging.json', {'manifest_sha256': ACCEPTED_MANIFEST_SHA256,
    'topology_files': 68, 'topology_bytes': sum(row['bytes'] for row in topology),
    'train_rows': 3378606, 'development_rows': 0, 'geometry_opened': False})
""")
    add("code", "cpu-observation", """# 复用实际数据加载器产生全量train-only统计，供后续冻结新全量recipe。
if MODE == 'prepare':
    worker = r'''
import json, os, time
from pathlib import Path
import torch
from molgap.family_training_data import DatasetSelection, load_training_data
from molgap.training_reproducibility import atomic_json
torch.set_num_threads(4)
started, cpu = time.perf_counter(), time.process_time()
data = load_training_data(Path(os.environ['FULL_DATA']), DatasetSelection('manifest.json', os.environ['FULL_PIN'], False))
record = {'source_commit': os.environ['SOURCE_PIN'], 'manifest_sha256': data.manifest_sha256,
    'train_rows': data.train_rows, 'development_rows': data.development_rows,
    'target_mean_eV': data.target_mean_eV, 'target_std_eV': data.target_std_eV,
    'wall_seconds': time.perf_counter()-started, 'process_cpu_seconds': time.process_time()-cpu,
    'training_executed': False, 'inference_executed': False, 'device': 'cpu', 'torch': torch.__version__}
atomic_json(Path(os.environ['OBSERVATION']), record)
print(json.dumps(record), flush=True)
'''
    subprocess.run([PYTHON, '-c', worker], check=True,
        env=dict(env, CUDA_VISIBLE_DEVICES='-1', FULL_DATA=str(DATA_ROOT), FULL_PIN=ACCEPTED_MANIFEST_SHA256,
            SOURCE_PIN=SOURCE_COMMIT, OBSERVATION=str(RUN_ROOT / 'full_cache_observation.json')))
    print('CPU preparation complete. Freeze both full recipes from the saved observation before using train mode.')
""")
    add("markdown", "training-notes", """## A100 正式阶段

仅在 `MODE='train'` 执行。`RELEASE_ROOT/<arm>/` 必须由已有 `prepare-family-training` 生成，包含
`prepared_training.json`、`package/`、`inputs/`、无错误的 `release_report.json`。
每份必须是一个单臂 Colab/A100 全量任务，数据无 development；K1 final-live、GPTrans final-ema、步数和学习率均由冻结recipe决定。

下方只做平台适配：读取既有prepared报告，构造worker需要的launch，调用原生preflight/train。不会修改recipe。
长任务在会话墙限到达前向worker发送SIGTERM，由已有训练器原子保存PAUSED状态；重新运行同任务时验证并恢复完整状态。
所有正式输出直接落Drive。若进程异常中断导致文件损坏，保留现场交给原验收/恢复入口，不覆盖重试。
""")
    add("code", "a100-run", """if MODE == 'train':
    from threading import Timer
    from molgap.experiment_package import verify_experiment_source_package
    from molgap.experiment_allocation import AllocationLedger
    allocation_boundary = SESSION_STARTED
    for arm in ARM_ORDER:
        prepared = RELEASE_ROOT / arm
        report = json.loads((prepared / 'prepared_training.json').read_text())
        release = json.loads((prepared / 'release_report.json').read_text())
        assert report['status'] == 'SOURCE_INPUTS_PREPARED' and not release['errors']
        assert len(report['jobs']) == 1 and report['jobs'][0]['arm_id'] == arm
        package = prepared / 'package'
        verified = verify_experiment_source_package(package)
        assert verified['package_identity'] == report['expected_package_identity']
        spec = json.loads((package / 'experiment_spec.json').read_text())
        assert spec['platform']['name'] == 'colab' and 'A100' in spec['platform']['accelerator']
        assert spec['platform']['device_count'] == 1
        source_root = Path('/content/molgap-full-source-' + arm)
        source_root.mkdir(exist_ok=True)
        with tarfile.open(package / 'source.tar.gz', 'r:gz') as frozen:
            frozen.extractall(source_root, filter='data')
        job = report['jobs'][0]
        recipe = json.loads((source_root / job['recipe']).read_text())
        assert recipe['train_rows'] == 3378606 and recipe['development_rows'] == 0
        assert recipe['dataset']['manifest_sha256'] == ACCEPTED_MANIFEST_SHA256
        assert recipe['dataset']['development'] is False
        assert recipe['logging'] == {'mode': 'epoch', 'interval': None}
        if arm == 'k1_muon':
            assert recipe['family'] == 'neural_atom_k1' and recipe['mode'] == 'reference'
            assert recipe['optimizer'] == 'k1-muon-batched-adamw-fp32-v1'
            assert recipe['selection'] == 'final-live'
        else:
            assert recipe['family'] == 'gptrans_t'
            assert recipe['mode'] == 'degree_bond_local_pair_update_norm_ema999'
            assert recipe['selection'] == 'final-ema'
        staged = prepared / 'inputs'
        launch = {'jobs': report['jobs'], 'expected_package_identity': report['expected_package_identity'],
            'prospective_sha256': report['prospective_sha256'], 'account': 'desktop-colab',
            'run_reference': 'desktop-colab/' + spec['logical_run_id']}
        launch_path = staged / 'colab_launch.json'
        if launch_path.exists():
            assert json.loads(launch_path.read_text()) == launch, 'Preserve the original launch identity'
        else:
            atomic_json(launch_path, launch)
        output = RUN_ROOT / arm
        preflight = RUN_ROOT / (arm + '-preflight-' + str(time.time_ns()))
        worker_env = dict(env, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(source_root / 'src'))
        common = [PYTHON, '-u', '-m', 'molgap.experiment_training_worker',
            '--source-root', str(source_root), '--package-dir', str(package), '--input-root', str(DATA_ROOT),
            '--launch', str(launch_path), '--arm', arm]
        if (output / 'output_manifest.json').is_file():
            print('Already has a retained terminal manifest:', arm, '— keep it for acceptance.')
            continue
        active_allocation_root = RUN_ROOT / 'allocation' / arm / str(time.time_ns())
        active_allocation = AllocationLedger(spec_identity=report['spec_identity'],
            hardware=observed_gpu, assignments={arm: 0}, started=allocation_boundary)
        active_allocation.write(active_allocation_root, 'running')
        remaining = SESSION_SECONDS - (time.perf_counter()-SESSION_STARTED) - CLEANUP_SECONDS
        assert remaining > CLEANUP_SECONDS, 'Resume in a new session; insufficient remaining wall budget'
        preflight_cmd = common + ['--phase', 'preflight', '--output', str(preflight)]
        if (output / 'last_checkpoint.pt').is_file():
            preflight_cmd += ['--resume-output', str(output)]
        subprocess.run(preflight_cmd, env=worker_env, check=True, timeout=remaining)
        remaining = SESSION_SECONDS - (time.perf_counter()-SESSION_STARTED) - CLEANUP_SECONDS
        assert remaining > 0
        process = subprocess.Popen(common + ['--phase', 'train', '--output', str(output),
            '--preflight-output', str(preflight)], env=worker_env)
        pause_timer = Timer(remaining, lambda: process.send_signal(signal.SIGTERM))
        pause_timer.start()
        try:
            returncode = process.wait(timeout=remaining+CLEANUP_SECONDS)
        finally:
            pause_timer.cancel()
        assert returncode == 0, f'Worker exit {returncode}; preserve {output}'
        allocated = active_allocation.write(active_allocation_root, 'complete')
        allocation_boundary = active_allocation.started + allocated['wall_seconds']
        active_allocation = None
        if not (output / 'output_manifest.json').is_file():
            assert (output / 'pause_receipt.json').is_file(), 'Missing terminal and pause records'
            print('Durable PAUSED state retained. Reconnect and resume the same frozen task:', arm)
            break
    print('Colab execution ended; scientific/RML acceptance is a separate existing local operation.')
""")
    add("code", "release-runtime", """# 成功收尾；错误时首单元注册的hook也会调用同一个Colab官方接口。
print('Durable records:', RUN_ROOT, flush=True)
if AUTO_RELEASE:
    runtime.unassign()
""")
    add("markdown", "links", """完成后核对Colab已断开并删除运行时，避免空闲计费。
自动释放使用Google Colab官方 [runtime.unassign](https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py)。
本地后续验收复用已有family输出、V5/RML接口；不要下载整份训练目录。
""")
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata={
        'colab': {'name': 'MolGap_K1_Muon_GPTrans_Full_A100.ipynb'},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}})
    output = HERE / 'MolGap_K1_Muon_GPTrans_Full_A100.ipynb'
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    exported = EXPORT / output.name
    shutil_output = output.read_bytes()
    exported.write_bytes(shutil_output)
    print(json.dumps({'notebook': str(exported), 'bytes': len(shutil_output),
        'sha256': hashlib.sha256(shutil_output).hexdigest(), 'code_cells_ast_valid': True,
        'executed_cloud': False, 'training_started': False}))


if __name__ == '__main__':
    main()
