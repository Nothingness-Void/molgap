"""CPU-only regression tests for the Modern2D native bootstrap repair.

The entrypoint is loaded from its real platform file.  Its source archive
contains only tiny stdlib fixtures, and every hardware/timer/runtime boundary
is replaced with a local fake.  No model, torch, GPU, network, or molecular
payload is imported or opened.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ENTRY_PATH = ROOT / "platforms/kaggle/run_modern2d_qualification.py"
PREP_PATH = ROOT / "src/molgap/modern2d_qualification_prepare.py"
HELPER_PATH = ROOT / "src/molgap/kaggle_mounts.py"
ONE_MIB = 1024 * 1024


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tar_bytes(members: list[tuple[str, bytes]]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as bundle:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


class _SignalFake:
    SIGALRM = object()
    ITIMER_REAL = object()

    def __init__(self, *, cleanup_failure: BaseException | None = None,
                 restore_failure: BaseException | None = None) -> None:
        self.cleanup_failure = cleanup_failure
        self.restore_failure = restore_failure
        self.calls: list[tuple[str, object]] = []
        self._previous = object()

    def signal(self, signal_number, handler):
        self.calls.append(("signal", handler))
        if handler is self._previous and self.restore_failure is not None:
            raise self.restore_failure
        if handler is self._previous:
            return None
        return self._previous

    def setitimer(self, timer, seconds):
        self.calls.append(("setitimer", seconds))
        if seconds == 0 and self.cleanup_failure is not None:
            raise self.cleanup_failure


class _BootstrapCase:
    def __init__(self, root: Path, *, layout: str = "flat",
                 helper_source: bytes | None = None,
                 archive_helper_source: bytes | None = None,
                 duplicate_helper: bool = False,
                 runtime_source: bytes | None = None,
                 oversized_launch: bool = False) -> None:
        self.root = root
        self.input_root = root / "input"
        self.output_root = root / "working"
        self.input_root.mkdir()
        self.output_root.mkdir()
        if layout == "flat":
            self.source_root = self.input_root / "qualification-source"
            self.source_relative = "qualification-source"
        elif layout == "owner_namespaced":
            self.source_root = self.input_root / "datasets" / "owner" / "qualification-source"
            self.source_relative = "datasets/owner/qualification-source"
        else:
            raise AssertionError(layout)
        self.source_root.mkdir(parents=True)

        self.helper_source = (helper_source if helper_source is not None
                              else HELPER_PATH.read_bytes())
        archive_helper = (archive_helper_source if archive_helper_source is not None
                          else self.helper_source)
        sentinel = root / "runtime-imported.txt"
        if runtime_source is None:
            runtime_source = (
                "from pathlib import Path\n"
                f"Path({str(sentinel)!r}).write_text('imported', encoding='utf-8')\n"
                "def bootstrap(**kwargs):\n"
                "    return {'status': 'complete', 'hardware': ['Tesla T4', 'Tesla T4']}\n"
            ).encode("utf-8")
        members = [
            ("src/molgap/kaggle_mounts.py", archive_helper),
            ("src/molgap/modern2d_qualification_runtime.py", runtime_source),
        ]
        if duplicate_helper:
            members.append(("src/molgap/kaggle_mounts.py", archive_helper))
        archive = _tar_bytes(members)
        self.archive = self.source_root / "source_payload.bin"
        self.archive.write_bytes(archive)
        source_digest = _sha_bytes(archive)

        if oversized_launch:
            launch_bytes = b"{" + b"x" * ONE_MIB + b"}"
        else:
            launch_bytes = json.dumps(
                {"expected_source_archive_sha256": source_digest},
                sort_keys=True,
            ).encode("utf-8")
        self.launch = self.source_root / "modern2d_qualification_launch.json"
        self.launch.write_bytes(launch_bytes)
        self.launch_digest = _sha_bytes(launch_bytes)
        self.sentinel = sentinel
        self.archive_digest = source_digest

    def patch_pathlib(self, entry):
        real_path = Path

        def redirect(raw):
            if str(raw) == "/kaggle/input":
                return self.input_root
            if str(raw) == "/kaggle/working":
                return self.output_root
            return real_path(raw)

        # The prepared helper is executed from the entry source string.  Give
        # that isolated code the same local mount root without changing the
        # real entry module's already-bound Path class.
        pathlib_proxy = types.ModuleType("pathlib")
        pathlib_proxy.Path = redirect
        return mock.patch.object(entry, "Path", redirect), \
            mock.patch.dict(sys.modules, {"pathlib": pathlib_proxy})


class Modern2DBootstrapRepairTests(unittest.TestCase):
    def _run_entry(self, case: _BootstrapCase, *, hardware=None,
                   hardware_failure: BaseException | None = None,
                   signal_fake: _SignalFake | None = None,
                   expected_launch: str | None = None):
        entry = _load(ENTRY_PATH, f"_test_modern2d_entry_{id(case)}")
        entry.EXPECTED_LAUNCH_SHA256 = expected_launch or case.launch_digest
        entry.EXPECTED_SOURCE_DATASET = "owner/qualification-source"
        entry.MOUNT_RESOLVER_SOURCE = case.helper_source.decode("utf-8")
        if signal_fake is None:
            signal_fake = _SignalFake()
        if hardware is None:
            hardware = "Tesla T4\nTesla T4\n"

        def check_output(*args, **kwargs):
            if hardware_failure is not None:
                raise hardware_failure
            return hardware

        path_patch, pathlib_patch = case.patch_pathlib(entry)
        with path_patch, pathlib_patch, \
                mock.patch.object(entry, "signal", signal_fake), \
                mock.patch.object(entry.subprocess, "check_output", check_output):
            return entry.main(), entry, signal_fake

    @staticmethod
    def _report(case: _BootstrapCase) -> dict:
        report = case.output_root / "bootstrap_allocation.json"
        if not report.is_file():
            raise AssertionError("bootstrap report was not retained")
        return json.loads(report.read_text(encoding="utf-8"))

    def test_real_entry_resolves_flat_and_namespaced_source_before_runtime_import(self):
        for layout in ("flat", "owner_namespaced"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as temporary:
                case = _BootstrapCase(Path(temporary), layout=layout)
                result, _, _ = self._run_entry(case)
                self.assertIsNone(result)
                self.assertTrue(case.sentinel.is_file(), "runtime was not reached")
                report = self._report(case)
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["phase"], "runtime_returned")
                self.assertEqual(report["source_discovery"]["status"], "selected")
                self.assertEqual(report["source_discovery"]["candidate_count"], 1)
                self.assertEqual(report["source_discovery"]["selected_root"], case.source_relative)
                self.assertEqual(len(report["source_discovery"]["candidates"]), 2)
                self.assertEqual(report["allocation_device_count"], 2)
                self.assertEqual(report["cost_status"], "PYTHON_OBSERVED_WINDOW_ONLY")
                self.assertIsNone(report["full_allocation_seconds"])
                self.assertIsNone(report["full_allocation_device_seconds"])
                self.assertIsNone(report["full_allocation_cost"])
                self.assertEqual(report["python_observed_device_seconds"],
                                 2 * report["python_observed_wall_seconds"])
                self.assertFalse((case.output_root / "bootstrap_allocation.tmp").exists())

    def test_source_mount_failure_is_retained_before_frozen_runtime_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = _BootstrapCase(Path(temporary))
            case.launch.unlink()
            with self.assertRaises(ValueError):
                self._run_entry(case)
            report = self._report(case)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["phase"], "source_discovery")
            self.assertEqual(report["source_discovery"]["status"], "rejected")
            self.assertFalse(case.sentinel.exists())
            self.assertIsNone(report["full_allocation_seconds"])

    def test_launch_content_tamper_fails_after_bounded_source_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = _BootstrapCase(Path(temporary))
            original_digest = case.launch_digest
            case.launch.write_bytes(case.launch.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "launch differs"):
                self._run_entry(case, expected_launch=original_digest)
            report = self._report(case)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["phase"], "launch_verification")
            self.assertIn("launch differs", report["failure"])
            self.assertFalse(case.sentinel.exists())

    def test_launch_hash_binding_rejects_wrong_expected_digest_without_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = _BootstrapCase(Path(temporary))
            with self.assertRaisesRegex(ValueError, "launch differs"):
                self._run_entry(case, expected_launch="0" * 64)
            report = self._report(case)
            self.assertEqual(report["phase"], "launch_verification")
            self.assertFalse(case.sentinel.exists())

    def test_launch_marker_read_is_bounded_to_one_mib_plus_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = _BootstrapCase(Path(temporary), oversized_launch=True)
            read_sizes: list[int] = []
            real_open = Path.open

            class _ReadSpy:
                def __init__(self, stream):
                    self.stream = stream

                def __enter__(self):
                    self.stream.__enter__()
                    return self

                def __exit__(self, *args):
                    return self.stream.__exit__(*args)

                def read(self, size=-1):
                    read_sizes.append(size)
                    return self.stream.read(size)

                def __getattr__(self, name):
                    return getattr(self.stream, name)

            def wrapped_open(path, *args, **kwargs):
                stream = real_open(path, *args, **kwargs)
                if path == case.launch:
                    return _ReadSpy(stream)
                return stream

            entry = _load(ENTRY_PATH, f"_test_modern2d_entry_large_{id(case)}")
            entry.EXPECTED_LAUNCH_SHA256 = case.launch_digest
            entry.EXPECTED_SOURCE_DATASET = "owner/qualification-source"
            entry.MOUNT_RESOLVER_SOURCE = case.helper_source.decode("utf-8")
            signal_fake = _SignalFake()
            path_patch, pathlib_patch = case.patch_pathlib(entry)
            with path_patch, pathlib_patch, \
                    mock.patch.object(entry, "signal", signal_fake), \
                    mock.patch.object(entry.subprocess, "check_output", return_value="Tesla T4\n"), \
                    mock.patch.object(Path, "open", wrapped_open):
                with self.assertRaisesRegex(ValueError, "exceeds one MiB"):
                    entry.main()
            self.assertEqual(read_sizes, [ONE_MIB + 1])
            report = self._report(case)
            self.assertEqual(report["phase"], "launch_verification")
            self.assertFalse(case.sentinel.exists())

    def test_archive_helper_mismatch_is_rejected_before_runtime_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            expected = HELPER_PATH.read_bytes()
            case = _BootstrapCase(Path(temporary), archive_helper_source=expected + b"# tampered\n")
            with self.assertRaisesRegex(ValueError, "mount resolver differs"):
                self._run_entry(case)
            report = self._report(case)
            self.assertEqual(report["phase"], "resolver_source_verification")
            self.assertIn("mount resolver differs", report["failure"])
            self.assertFalse(case.sentinel.exists())

    def test_oversized_archive_helper_is_rejected_without_unbounded_extract(self):
        with tempfile.TemporaryDirectory() as temporary:
            oversized = HELPER_PATH.read_bytes() + b"x" * (ONE_MIB + 1)
            case = _BootstrapCase(Path(temporary), archive_helper_source=oversized)
            with self.assertRaises(ValueError):
                self._run_entry(case)
            report = self._report(case)
            self.assertEqual(report["phase"], "resolver_source_verification")
            self.assertFalse(case.sentinel.exists())

    def test_hardware_failure_retains_unknown_full_cost_and_failure_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = _BootstrapCase(Path(temporary))
            failure = RuntimeError("fake hardware probe failed")
            with self.assertRaisesRegex(RuntimeError, "fake hardware probe failed"):
                self._run_entry(case, hardware_failure=failure)
            report = self._report(case)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["phase"], "hardware_observation")
            self.assertIn("fake hardware probe failed", report["failure"])
            self.assertEqual(report["cost_status"], "PYTHON_OBSERVED_WINDOW_ONLY")
            self.assertIsNone(report["full_allocation_seconds"])
            self.assertIsNone(report["full_allocation_device_seconds"])
            self.assertIsNone(report["full_allocation_cost"])
            self.assertIsNone(report["allocation_device_count"])
            self.assertIsNone(report["python_observed_device_seconds"])
            self.assertFalse(case.sentinel.exists())

    def test_timer_cleanup_failure_on_normal_return_is_nonzero_after_atomic_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = _BootstrapCase(Path(temporary))
            signal_fake = _SignalFake(cleanup_failure=OSError("timer disarm failed"))
            with self.assertRaisesRegex(RuntimeError, "deadline cleanup failed"):
                self._run_entry(case, signal_fake=signal_fake)
            report = self._report(case)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(report["deadline_cleanup_errors"])
            self.assertEqual(report["cost_status"], "PYTHON_OBSERVED_WINDOW_ONLY")
            self.assertFalse((case.output_root / "bootstrap_allocation.tmp").exists())

    def test_timer_cleanup_failure_does_not_mask_original_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = _BootstrapCase(Path(temporary))
            signal_fake = _SignalFake(cleanup_failure=OSError("timer disarm failed"))
            original = RuntimeError("original hardware failure")
            with self.assertRaisesRegex(RuntimeError, "original hardware failure"):
                self._run_entry(case, hardware_failure=original, signal_fake=signal_fake)
            report = self._report(case)
            self.assertEqual(report["status"], "failed")
            self.assertIn("original hardware failure", report["failure"])
            self.assertTrue(report["deadline_cleanup_errors"])

    def test_prepare_package_injects_launch_dataset_and_packaged_helper_source(self):
        # This imports only the preparation/lifecycle boundary.  All model,
        # package, data, and RML owners are replaced by tiny stdlib fakes.
        # Import through the real package name so its relative lifecycle
        # imports remain intact; the imported owners below are still faked.
        import importlib
        owner = importlib.import_module("molgap.modern2d_qualification_prepare")
        helper_source = HELPER_PATH.read_text(encoding="utf-8")
        entry_source = ENTRY_PATH.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            output = Path(temporary) / "prepared"
            mounted = Path(temporary) / "mounted"
            root.mkdir()
            mounted.mkdir()
            here = root / owner.BASE
            here.mkdir(parents=True)
            authority = {
                "purpose": "EXECUTION_QUALIFICATION_ONLY", "user_authorized": True,
                "arms": list(owner.ARMS), "seeds": [42], "physical_batch_size": 128,
                "gpu": {"platform": "kaggle2", "owner": owner.ACCOUNT,
                        "accelerator": "NvidiaTeslaT4", "allocated_devices": 2,
                        "wall_ceiling_seconds": 1800, "allocated_t4_hours_ceiling": 1},
                "scientific_training_authorized": False,
                "protected_role_access_authorized": False,
                "extra_seed_authorized": False,
                "automatic_successor_authorized": False,
            }
            (here / "authority.json").write_text(json.dumps(authority), encoding="utf-8")
            metadata = {"training_input": {"path": "training_input.json", "sha256": ""},
                        "export_acceptance_sha256": ""}
            training_input = here / "training_input.json"
            training_input.write_text("{}", encoding="utf-8")
            metadata["training_input"]["sha256"] = _sha_bytes(training_input.read_bytes())
            metadata_path = here / "qualification_inputs.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            metadata_sha = _sha_bytes(metadata_path.read_bytes())
            acceptance = mounted / owner.MOUNT / "indexed_assets" / "acceptance.json"
            acceptance.parent.mkdir(parents=True)
            acceptance.write_text(json.dumps({
                "roles": {
                    "train_100k": {"logical_content_sha256": "5" * 64},
                    "common_dev": {"logical_content_sha256": "6" * 64},
                }
            }), encoding="utf-8")
            metadata["export_acceptance_sha256"] = _sha_bytes(acceptance.read_bytes())
            binding = {
                "train": {"permit": {"version_identity": "4" * 64}},
                "development": {"permit": {"version_identity": "4" * 64}},
            }
            training_input.write_text(json.dumps(binding), encoding="utf-8")
            (acceptance.parent.parent / "training_input.json").write_bytes(
                training_input.read_bytes()
            )
            metadata["training_input"]["sha256"] = _sha_bytes(training_input.read_bytes())
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            metadata_sha = _sha_bytes(metadata_path.read_bytes())
            prep = {"spec_identity": "s" * 64, "metadata_sha256": metadata_sha,
                    "source_records": [], "jobs": []}
            (here / "experiment_spec.json").write_text(json.dumps({"identity": "s" * 64}), encoding="utf-8")
            (here / "preparation.json").write_text(json.dumps(prep), encoding="utf-8")

            fake_modules = {}
            spec_module = types.ModuleType("molgap.experiment_spec")

            class FakeSpec:
                def __init__(self, value):
                    self.identity = value["identity"]

                def to_dict(self):
                    return {"identity": self.identity}

            spec_module.ExperimentSpec = FakeSpec
            spec_module.TERMINAL_PROTOCOL = {}
            fake_modules["molgap.experiment_spec"] = spec_module

            package_module = types.ModuleType("molgap.experiment_package")

            def build_package(spec, repo_root, sources, package):
                package.mkdir(parents=True)
                return {"package_identity": "p" * 64, "archive_sha256": "a" * 64,
                        "source_commit": "c" * 40}

            def stage_inputs(package, staged, states):
                staged.mkdir(parents=True)
                return staged

            package_module.build_experiment_source_package = build_package
            package_module.stage_source_inputs = stage_inputs
            package_module.read_packaged_text = lambda package, name: (
                entry_source if name == owner.ENTRY else helper_source)
            fake_modules["molgap.experiment_package"] = package_module

            preflight_module = types.ModuleType("molgap.experiment_preflight")
            preflight_module._unpack = lambda package, source: source.mkdir(parents=True, exist_ok=True)
            preflight_module.check_release_inputs = lambda *args, **kwargs: {
                "status": "LOCAL_RELEASE_INPUTS_VERIFIED"}
            fake_modules["molgap.experiment_preflight"] = preflight_module

            inventory_module = types.ModuleType("molgap.experiment_source_inventory")
            inventory_module.runtime_source_files = lambda *args, **kwargs: []
            fake_modules["molgap.experiment_source_inventory"] = inventory_module

            prospective_module = types.ModuleType("molgap.experiment_prospective")
            prospective_module._prospective_inputs = lambda *args, **kwargs: ([], None)
            fake_modules["molgap.experiment_prospective"] = prospective_module

            workflow_module = types.ModuleType("molgap.experiment_family_workflow")
            workflow_module.RunContext = object
            fake_modules["molgap.experiment_family_workflow"] = workflow_module

            qualification_module = types.ModuleType("molgap.modern2d_qualification")
            qualification_module.Modern2DQualificationContext = object
            fake_modules["molgap.modern2d_qualification"] = qualification_module

            runtime_module = types.ModuleType("molgap.modern2d_qualification_runtime")
            runtime_module.LAUNCH_FORMAT = "fixture-launch"
            runtime_module.LAUNCH_FILENAME = "modern2d_qualification_launch.json"
            runtime_module.validate_launch = lambda **kwargs: None
            fake_modules["molgap.modern2d_qualification_runtime"] = runtime_module

            data_module = types.ModuleType("molgap.data.indexed_inputs")
            data_module.AcceptedIndexedInputPermit = object
            fake_modules["molgap.data.indexed_inputs"] = data_module
            runner_module = types.ModuleType("molgap.family_training_runner")
            runner_module.read_recipe = lambda *args, **kwargs: (None, None)
            fake_modules["molgap.family_training_runner"] = runner_module
            plan_module = types.ModuleType("molgap.research_memory.plan")
            plan_module.plan_many = lambda *args, **kwargs: {"status": "PLANNED"}
            fake_modules["molgap.research_memory.plan"] = plan_module

            with mock.patch.dict(sys.modules, fake_modules):
                report = owner.prepare_package(
                    root, output, init_root=Path(temporary) / "initial",
                    mounted_root=mounted, expected_spec_identity="s" * 64)

            packaged_entry = (output / "run_modern2d_qualification.py").read_text(encoding="utf-8")
            self.assertIn("EXPECTED_LAUNCH_SHA256 = ", packaged_entry)
            self.assertIn(f"EXPECTED_SOURCE_DATASET = {owner.SOURCE_DATASET!r}", packaged_entry)
            self.assertIn(f"MOUNT_RESOLVER_SOURCE = {helper_source!r}", packaged_entry)
            self.assertEqual(report["status"], "QUALIFICATION_PACKAGE_PREPARED")


if __name__ == "__main__":
    unittest.main()
