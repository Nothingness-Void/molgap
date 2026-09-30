"""Per-arm wiring of existing EdgeState primitives, not a new training recipe."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from . import edge_state_training_core as core
from .experiment_package import _spec, _source, verify_experiment_source_package
from .experiment_spec import SCHEMA_VERSION_V2, ExperimentSpec
from .research_memory.trace import RMLTraceRecorder, atomic_write, file_digest, json_bytes


@dataclass(frozen=True)
class EdgeStateScreen:
    spec: ExperimentSpec
    binding: core.EdgeStateTrainingBinding
    trajectory_id: str
    run_id: str
    output: Path

    def recorder(self, *, metric_semantics: dict, device_time_semantics: str | None = None) -> RMLTraceRecorder:
        # Metrics/role semantics belong to the executable contract, not a guess here.
        return RMLTraceRecorder(self.output / "canonical_trace.json",
                               trajectory_id=self.trajectory_id, run_id=self.run_id,
                               metric_semantics=metric_semantics, device_time_semantics=device_time_semantics)

    def sampler(self, *, epoch: int, start_batch: int = 0):
        return core.EpochPermutationBatchSampler(
            self.binding.train_rows, self.binding.physical_batch_size,
            seed=self.binding.initialization_seed, epoch=epoch, start_batch=start_batch)

    def construct_model(self):
        return core.construct_bound_model(self.spec, self.binding)

    def step(self, model, optimizer, batch, *, clip_norm: float):
        return core.normalized_gap_step(model, optimizer, batch, self.binding, clip_norm=clip_norm)

    def evaluate(self, model, batches, *, expected_source_idx):
        return core.evaluate_development(model, batches, self.binding, expected_source_idx=expected_source_idx)

    def save_checkpoint(self, model, optimizer, scheduler, **cursor):
        return core.save_checkpoint(self.output / "last.pt", self.binding, model, optimizer, scheduler, **cursor)

    def restore_checkpoint(self, model, optimizer, scheduler, *, expected_sha256: str, loader_generator=None):
        return core.restore_checkpoint(
            self.output / "last.pt", expected_sha256, self.binding, model, optimizer, scheduler,
            loader_generator=loader_generator)


def bind_edge_state_screen(
    spec, arm_id: str, *, package_dir: Path, output_root: Path,
    graph_manifest: Path, expected_graph_manifest_sha256: str,
    training_contract: Path, expected_training_contract_sha256: str,
    runtime_fingerprint: str, train_rows: int, target_mean_eV: float, target_std_eV: float,
) -> EdgeStateScreen:
    """Bind a base/depth/K1 arm to verified metadata and a distinct output root.

    The caller still authenticates graph roles, the executable contract,
    transform statistics, runtime certificate and release authority. The
    model-only registry recipe is NOT promoted to executable by this helper.
    Existing historical K1 trainers are not redirected to this factory.
    """
    spec = _spec(spec)
    declaration = spec.to_dict()
    if declaration["schema_version"] != SCHEMA_VERSION_V2:
        raise ValueError("EdgeState screen requires per-arm Spec v2 plans")
    package = verify_experiment_source_package(package_dir)
    if package["spec_identity"] != spec.identity:
        raise ValueError("EdgeState source package differs from the arm Spec")
    for path, expected in ((graph_manifest, expected_graph_manifest_sha256),
                           (training_contract, expected_training_contract_sha256)):
        path = Path(path).absolute()
        _source(path.parent.resolve(), path.name)
        if file_digest(path) != expected:
            raise ValueError("EdgeState metadata artifact SHA mismatch")
    binding = core.EdgeStateTrainingBinding.from_spec(
        spec, arm_id, source_commit=package["source_commit"],
        source_package_sha256=package["archive_sha256"],
        graph_manifest_sha256=expected_graph_manifest_sha256,
        training_contract_sha256=expected_training_contract_sha256,
        runtime_fingerprint=runtime_fingerprint, train_rows=train_rows, physical_batch_size=128,
        target_mean_eV=target_mean_eV, target_std_eV=target_std_eV)
    plan = next(item for item in declaration["prospective"]["arms"] if item["arm_id"] == arm_id)
    output = Path(output_root).absolute() / arm_id
    # Check the whole path for links before publishing the immutable binding.
    from .experiment_launch import _safe_local
    _safe_local(output)
    record = {"binding": asdict(binding), "trajectory_id": plan["trajectory_id"],
              "run_id": f"{declaration['logical_run_id']}:{arm_id}"}
    path = output / "screen_binding.json"
    if path.exists():
        if path.read_bytes() != json_bytes(record):
            raise ValueError("EdgeState screen/resume identity changed")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Existing outputs without a screen binding require reconciliation")
    else:
        atomic_write(path, json_bytes(record))
    return EdgeStateScreen(spec, binding, record["trajectory_id"], record["run_id"], output)
