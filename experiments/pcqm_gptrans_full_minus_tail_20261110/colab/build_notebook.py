"""Build the GPT-T minus-tail Colab launcher (A100 or L4); never execute it locally."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIMARY = Path("D:/文档/molgap")
EXPORT = Path("D:/下载/MolGap-Colab-MinusTail-20261110")
SOURCE_COMMIT = "aad49f40f8e2500e6f52e57ddb838bd35c7bdd6e"


def main():
    EXPORT.mkdir(parents=True, exist_ok=True)
    manifest_sha = '100716d0355e672d42fe50f693c096f5651b75c72fd5ab93667884434d570e5d'
    releases = json.loads((HERE / 'release_assets.json').read_text())
    archives = json.loads((HERE / 'data_archives.json').read_text())
    cells = []
    def add(kind, name, text):
        cell = dict(cell_type=kind, id=name, metadata={}, source=text.splitlines(True))
        if kind == "code":
            ast.parse(text)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "intro", """# MolGap：GPT-T（排除尾部保留集）全量 Colab

**选择 A100 或 L4 后运行全部单元。训练包与 TRAIN 数据需先放入 MolGap 云盘。**

1. 挂载 Google Drive，自动核验并解包一份 `prepare-family-training` 产物。
2. 单张 A100 或 L4 训练 GPTrans BondLocal＋PairNorm（B128，冻结配方），只用 TRAIN 前 3,250,000 行；后 128,606 行是保留集，不参与训练。
3. 60个epoch、1,523,400步、194,995,200次样本呈现。每epoch输出标准TRAIN日志。
4. 检查点、曲线、原生证明和输出保存到 Drive。单会话最多10小时，保存状态后释放运行时；再次运行全部单元精确续训。
5. 每完成3个epoch，把滚动检查点复制一份到 snapshots 目录，训练后离线评估保留集，只看曲线，不挑快照。
6. 正常结束、受控暂停或单元报错后自动释放。提前手动停止请使用“断开连接并删除运行时”。

只读取全量 TRAIN 拓扑，不读取几何、官方验证或测试。全量训练没有内部开发集，按冻结合同的 final-live/final-ema 输出；本笔记本不自行选择 epoch、改变学习率或补做官方评测。
""")
    add("code", "settings", """from pathlib import Path
import json, os, subprocess, sys, time, tarfile, zipfile, shutil, signal
from google.colab import runtime

# 先选择A100，再运行全部单元。每个新运行时从Drive重建本地环境。
AUTO_RELEASE = True
DRIVE_ROOT = Path('/content/drive/MyDrive/MolGap/V5')
FULL_CACHE_ON_DRIVE = DRIVE_ROOT / 'fixed_data/full_train_v1'
RELEASE_ROOT = DRIVE_ROOT / 'releases/gptrans-minus-tail-20261110'
RUN_ROOT = DRIVE_ROOT / 'runs/gptrans-minus-tail-20261110'
ARM_ORDER = ('gptrans_pairnorm',)
SNAPSHOT_EVERY_PASSES = 3
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
print('TRAIN minus tail: 3250000 rows | 60 passes | FP32 | A100 or L4')
""")
    add("code", "mount", """from google.colab import drive
drive.mount('/content/drive')
RUN_ROOT.mkdir(parents=True, exist_ok=True)
WORK_ROOT = Path('/content/molgap-run')
WORK_ROOT.mkdir(parents=True, exist_ok=True)
observed_gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip().splitlines()
assert len(observed_gpu) == 1 and any(name in observed_gpu[0] for name in ('A100', 'L4')), observed_gpu
print('GPU:', observed_gpu[0])
print('Outputs:', RUN_ROOT)
""")
    add("code", "frozen-source", f"""# 身份来自已准备的训练包；不在笔记本嵌入源码。
import hashlib
SOURCE_COMMIT = '{SOURCE_COMMIT}'
RELEASE_ASSETS = {repr(releases)}
ACCEPTED_MANIFEST_SHA256 = '{manifest_sha}'
manifest_bytes = (FULL_CACHE_ON_DRIVE / 'manifest_minus_tail_v1.json').read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest() == ACCEPTED_MANIFEST_SHA256
accepted_manifest = json.loads(manifest_bytes)
DATA_ARCHIVES = json.loads((FULL_CACHE_ON_DRIVE / 'archives/data_archives.json').read_text())
assert accepted_manifest['roles']['train']['rows'] == 3250000
assert accepted_manifest['roles']['development'] is None
""")
    add("code", "prepared-bundles", """# 只解包原始prepared文件；运行输出位于独立RUN_ROOT，不会被覆盖。
RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
for asset in RELEASE_ASSETS:
    archive = RELEASE_ROOT / asset['file']
    assert archive.is_file() and archive.stat().st_size == asset['bytes'], archive
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == asset['sha256'], archive
    with zipfile.ZipFile(archive) as bundle:
        assert all(name.startswith(asset['arm_id'] + '/') and '..' not in Path(name).parts
                   for name in bundle.namelist())
        bundle.extractall(RELEASE_ROOT)
    report = json.loads((RELEASE_ROOT / asset['arm_id'] / 'prepared_training.json').read_text())
    assert report['expected_package_identity'] == asset['package_identity']
    assert report['spec_identity'] == asset['spec_identity']
SOURCE_ROOT = Path('/content/molgap-colab-source-' + SOURCE_COMMIT[:8])
SOURCE_ROOT.mkdir(exist_ok=True)
bootstrap_package = RELEASE_ROOT / 'gptrans_pairnorm/package'
assert (bootstrap_package / 'SOURCE_COMMIT.txt').read_text().strip() == SOURCE_COMMIT
with tarfile.open(bootstrap_package / 'source.tar.gz', 'r:gz') as frozen:
    frozen.extractall(SOURCE_ROOT, filter='data')
sys.path.insert(0, str(SOURCE_ROOT / 'src'))
print('The frozen minus-tail prepared bundle is available.')
""")
    add("code", "environment", """# 使用独立环境，不改变Colab内核环境。
ENV_ROOT = Path('/content/molgap-full-train-py311')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'uv'], check=True)
subprocess.run([sys.executable, '-m', 'uv', 'venv', '--python', '3.11', '--seed', str(ENV_ROOT)], check=True)
PYTHON = str(ENV_ROOT / 'bin/python')
subprocess.run([PYTHON, '-m', 'pip', 'install', '--no-cache-dir', 'torch==2.11.0',
    '--index-url', 'https://download.pytorch.org/whl/cu128'], check=True)
subprocess.run([PYTHON, '-m', 'pip', 'install', '--no-cache-dir',
    'numpy<2', 'torch-geometric==2.6.1', 'ogb==1.3.6'], check=True)
env = dict(os.environ, PYTHONPATH=str(SOURCE_ROOT / 'src'), PYTHONUNBUFFERED='1',
    PYTHONHASHSEED='42', CUBLAS_WORKSPACE_CONFIG=':4096:8',
    OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
from molgap.training_reproducibility import sha256_file, atomic_json
print('Worker Python:', PYTHON)
""")
    add("code", "data-stage", """# 仅解压TRAIN拓扑；原始分片SHA由现有训练加载器统一核验。
topology = accepted_manifest['assets']['topology']
assert len(topology) == 65 and all(row['role'] == 'train' for row in topology)
assert sum(row['rows'] for row in topology) == 3250000
DATA_ROOT = Path('/content/molgap-full-data')
CACHE_ROOT = DATA_ROOT / 'ogb-train-full'
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
required = sum(row['bytes'] for row in topology if not (CACHE_ROOT / row['file']).is_file())
assert shutil.disk_usage('/content').free > required + 5 * 1024**3, 'Insufficient local SSD space'
records = {row['file']: row for row in topology}
assert set(records) <= {name for pack in DATA_ARCHIVES for name in pack['members']}
for number, pack in enumerate(DATA_ARCHIVES, 1):
    missing = [name for name in pack['members'] if name in records and not (CACHE_ROOT / name).is_file()]
    if missing:
        original = FULL_CACHE_ON_DRIVE / 'archives' / pack['file']
        assert original.is_file() and original.stat().st_size == pack['bytes'], original
        with zipfile.ZipFile(original) as bundle:
            assert sorted(bundle.namelist()) == sorted(pack['members'])
            for name in missing:
                bundle.extract(name, CACHE_ROOT)
    print('TRAIN topology archive', number, '/', len(DATA_ARCHIVES), flush=True)
(CACHE_ROOT / 'manifest.json').write_bytes(manifest_bytes)
atomic_json(RUN_ROOT / 'cache_staging.json', {'manifest_sha256': ACCEPTED_MANIFEST_SHA256,
    'topology_files': 65, 'topology_bytes': sum(row['bytes'] for row in topology),
    'train_rows': 3250000, 'development_rows': 0, 'geometry_opened': False})
""")
    add("markdown", "training-notes", """## A100 正式阶段

选择A100后执行。切换运行时会清空 `/content`，必须从头运行全部单元；源码、依赖和SSD缓存自动重建，训练检查点从Drive恢复。`RELEASE_ROOT/<arm>/` 必须由已有 `prepare-family-training` 生成，包含
`prepared_training.json`、`package/`、`inputs/`、无错误的 `release_report.json`。
每份必须是一个单臂 Colab/A100 全量任务，数据无 development；K1 final-live、GPTrans final-ema、步数和学习率均由冻结recipe决定。

下方只做平台适配：读取既有prepared报告，构造worker需要的launch，调用原生preflight/train。不会修改recipe。
长任务在会话墙限到达前向worker发送SIGTERM，由已有训练器原子保存PAUSED状态；重新运行同任务时验证并恢复完整状态。
所有正式输出直接落Drive。若进程异常中断导致文件损坏，保留现场交给原验收/恢复入口，不覆盖重试。
""")
    add("code", "a100-run", """from threading import Event, Thread, Timer


def copy_snapshot(source, directory, name):
    # Atomic copy of the trainer rolling checkpoint; never overwrites an earlier snapshot.
    target = directory / name
    if not source.is_file() or target.exists():
        return None
    partial = target.with_suffix('.partial')
    shutil.copyfile(source, partial)
    os.replace(partial, target)
    return target


def mirror(source, target):
    # Copy new or changed files; partial files are renamed into place so Drive never shows half a file.
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.name.startswith('.'):
            continue
        destination = target / path.relative_to(source)
        if destination.is_file() and destination.stat().st_size == path.stat().st_size and destination.stat().st_mtime >= path.stat().st_mtime:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + '.partial')
        shutil.copyfile(path, partial)
        os.replace(partial, destination)


def completed_passes(trace_path):
    try:
        return len(json.loads(trace_path.read_text())['observations'])
    except (OSError, ValueError, KeyError):
        return 0

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
    assert recipe['train_rows'] == 3250000 and recipe['development_rows'] == 0
    assert recipe['dataset']['manifest_sha256'] == ACCEPTED_MANIFEST_SHA256
    assert recipe['dataset']['development'] is False
    assert recipe['logging'] == {'mode': 'epoch', 'interval': None}
    assert arm == 'gptrans_pairnorm' and recipe['family'] == 'gptrans_t'
    assert recipe['mode'] == 'degree_bond_local_pair_update_norm_ema999' and recipe['selection'] == 'final-ema'
    staged = prepared / 'inputs'
    launch = {'jobs': report['jobs'], 'expected_package_identity': report['expected_package_identity'],
        'prospective_sha256': report['prospective_sha256'], 'account': 'desktop-colab',
        'run_reference': 'desktop-colab/' + spec['logical_run_id']}
    launch_path = staged / 'colab_launch.json'
    if launch_path.exists():
        assert json.loads(launch_path.read_text()) == launch, 'Preserve the original launch identity'
    else:
        atomic_json(launch_path, launch)
    # Drive cannot hard-link, which the native writers need: run on local disk, mirror to Drive.
    output = WORK_ROOT / arm
    drive_output = RUN_ROOT / arm
    if drive_output.is_dir() and not output.exists():
        mirror(drive_output, output)
    preflight = WORK_ROOT / (arm + '-preflight-' + str(time.time_ns()))
    worker_env = dict(env, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(source_root / 'src'))
    common = [PYTHON, '-u', '-m', 'molgap.experiment_training_worker',
        '--source-root', str(source_root), '--package-dir', str(package), '--input-root', str(DATA_ROOT),
        '--launch', str(launch_path), '--arm', arm]
    if (output / 'output_manifest.json').is_file():
        # The existing completed-run path inspects artifacts without training.
        subprocess.run(common + ['--phase', 'train', '--output', str(output)],
            env=worker_env, check=True)
        print('Existing terminal artifacts verified by the owning worker:', arm)
        continue
    active_allocation_root = WORK_ROOT / 'allocation' / arm / str(time.time_ns())
    active_allocation = AllocationLedger(spec_identity=report['spec_identity'],
        hardware=observed_gpu, assignments={arm: 0}, started=allocation_boundary)
    active_allocation.write(active_allocation_root, 'running')
    remaining = SESSION_SECONDS - (time.perf_counter()-SESSION_STARTED) - CLEANUP_SECONDS
    assert remaining > CLEANUP_SECONDS, 'Resume in a new session; insufficient remaining wall budget'
    preflight_cmd = common + ['--phase', 'preflight', '--output', str(preflight)]
    if (output / 'last_checkpoint.pt').is_file():
        preflight_cmd += ['--resume-output', str(output)]
    preflight_log = WORK_ROOT / (preflight.name + '.log')
    with preflight_log.open('w') as log:
        checked = subprocess.run(preflight_cmd, env=worker_env, stdout=log,
            stderr=subprocess.STDOUT, timeout=remaining)
    print(preflight_log.read_text(), flush=True)
    checked.check_returncode()
    remaining = SESSION_SECONDS - (time.perf_counter()-SESSION_STARTED) - CLEANUP_SECONDS
    assert remaining > 0
    process = subprocess.Popen(common + ['--phase', 'train', '--output', str(output),
        '--preflight-output', str(preflight)], env=worker_env)
    pause_timer = Timer(remaining, lambda: process.send_signal(signal.SIGTERM))
    pause_timer.start()
    snapshots = RUN_ROOT / (arm + '-snapshots')
    snapshots.mkdir(exist_ok=True)
    stop_snapshots = Event()
    def snapshot_loop():
        copied = set()
        mirrored = [0]
        while not stop_snapshots.wait(60):
            done = completed_passes(output / 'canonical_trace.json')
            if done and done != mirrored[0]:
                try:
                    mirror(output, drive_output)
                    mirrored[0] = done
                except OSError as error:
                    print('Mirror skipped:', error, flush=True)
            if done and done % SNAPSHOT_EVERY_PASSES == 0 and done not in copied:
                try:
                    print('Snapshot:', copy_snapshot(output / 'last_checkpoint.pt', snapshots, 'pass_%03d.pt' % done), flush=True)
                    copied.add(done)
                except OSError as error:
                    print('Snapshot skipped:', error, flush=True)
    Thread(target=snapshot_loop, daemon=True).start()
    try:
        returncode = process.wait(timeout=remaining+CLEANUP_SECONDS)
    finally:
        pause_timer.cancel()
        stop_snapshots.set()
        mirror(output, drive_output)
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
        'colab': {'name': 'MolGap_GPTrans_MinusTail.ipynb'},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}})
    output = HERE / 'MolGap_GPTrans_MinusTail.ipynb'
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    exported = EXPORT / output.name
    shutil_output = output.read_bytes()
    exported.write_bytes(shutil_output)
    print(json.dumps({'notebook': str(exported), 'bytes': len(shutil_output),
        'sha256': hashlib.sha256(shutil_output).hexdigest(), 'code_cells_ast_valid': True,
        'executed_cloud': False, 'training_started': False}))


if __name__ == '__main__':
    main()
