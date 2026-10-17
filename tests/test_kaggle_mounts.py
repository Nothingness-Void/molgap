"""Stdlib-only tests for the bounded Kaggle dataset mount resolver.

The resolver is deliberately exercised as a source file, rather than through
the model package, so these checks cannot import torch, enumerate a mounted
dataset, or read a marker's bytes.
"""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "src/molgap/kaggle_mounts.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("_test_kaggle_mounts", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KaggleMountResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = _load_helper()

    @staticmethod
    def _marker(root: Path, name: str = "modern2d_qualification_launch.json") -> Path:
        marker = root / name
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(b"marker payload must not be opened")
        return marker

    def test_flat_root_is_selected_and_report_has_three_bounded_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_root = Path(temporary)
            flat = input_root / "qualification-source"
            flat.mkdir()
            self._marker(flat)
            report = {"stale": "value"}

            with mock.patch.object(Path, "iterdir", side_effect=AssertionError("recursive enumeration")), \
                    mock.patch.object(Path, "rglob", side_effect=AssertionError("recursive enumeration")):
                selected = self.resolver.resolve_dataset_mount(
                    input_root, "owner/qualification-source",
                    marker="modern2d_qualification_launch.json", report=report)

            self.assertEqual(selected, flat)
            self.assertEqual(report["status"], "selected")
            self.assertEqual(report["candidate_count"], 1)
            self.assertEqual(report["selected_root"], "qualification-source")
            self.assertFalse(report["recursive_scan"])
            self.assertFalse(report["file_contents_read"])
            self.assertEqual(len(report["candidates"]), 3)
            self.assertEqual(
                {record["layout"] for record in report["candidates"]},
                {"flat", "owner_qualified", "owner_namespaced"},
            )
            self.assertEqual(
                {record["path"] for record in report["candidates"]},
                {
                    "qualification-source",
                    "owner/qualification-source",
                    "datasets/owner/qualification-source",
                },
            )

    def test_owner_namespaced_root_is_selected_when_flat_root_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_root = Path(temporary)
            namespaced = input_root / "datasets" / "owner" / "qualification-source"
            namespaced.mkdir(parents=True)
            self._marker(namespaced)
            report = {}

            selected = self.resolver.resolve_dataset_mount(
                input_root, "owner/qualification-source",
                mount_name="qualification-source",
                marker="modern2d_qualification_launch.json", report=report)

            self.assertEqual(selected, namespaced)
            self.assertEqual(report["candidate_count"], 1)
            self.assertEqual(report["selected_root"], "datasets/owner/qualification-source")
            by_layout = {record["layout"]: record for record in report["candidates"]}
            self.assertFalse(by_layout["flat"]["exists"])
            self.assertFalse(by_layout["owner_qualified"]["exists"])
            self.assertTrue(by_layout["owner_namespaced"]["marker_present"])

    def test_owner_qualified_root_is_selected_when_other_layouts_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_root = Path(temporary)
            owner_qualified = input_root / "owner" / "qualification-source"
            owner_qualified.mkdir(parents=True)
            self._marker(owner_qualified)
            report = {}

            selected = self.resolver.resolve_dataset_mount(
                input_root, "owner/qualification-source",
                marker="modern2d_qualification_launch.json", report=report)

            self.assertEqual(selected, owner_qualified)
            self.assertEqual(report["candidate_count"], 1)
            self.assertEqual(report["selected_root"], "owner/qualification-source")
            by_layout = {record["layout"]: record for record in report["candidates"]}
            self.assertTrue(by_layout["owner_qualified"]["marker_present"])
            self.assertFalse(by_layout["flat"]["exists"])
            self.assertFalse(by_layout["owner_namespaced"]["exists"])

    def test_flat_root_is_selected_when_namespaced_root_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_root = Path(temporary)
            flat = input_root / "qualification-source"
            flat.mkdir()
            self._marker(flat)

            report = {}
            selected = self.resolver.resolve_dataset_mount(
                input_root, "owner/qualification-source",
                marker="modern2d_qualification_launch.json", report=report)

            self.assertEqual(selected, flat)
            self.assertEqual(report["candidate_count"], 1)
            by_layout = {record["layout"]: record for record in report["candidates"]}
            self.assertTrue(by_layout["flat"]["marker_present"])
            self.assertFalse(by_layout["owner_qualified"]["exists"])
            self.assertFalse(by_layout["owner_namespaced"]["exists"])

    def test_duplicate_roots_are_rejected_even_when_only_one_has_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_root = Path(temporary)
            flat = input_root / "qualification-source"
            namespaced = input_root / "datasets" / "owner" / "qualification-source"
            flat.mkdir()
            namespaced.mkdir(parents=True)
            self._marker(flat)

            report = {}
            with self.assertRaisesRegex(ValueError, "exactly one dataset mount"):
                self.resolver.resolve_dataset_mount(
                    input_root, "owner/qualification-source",
                    marker="modern2d_qualification_launch.json", report=report)

            self.assertEqual(report["status"], "rejected")
            self.assertEqual(report["candidate_count"], 2)
            self.assertEqual(len(report["candidates"]), 3)
            by_layout = {record["layout"]: record for record in report["candidates"]}
            self.assertTrue(by_layout["flat"]["exists"])
            self.assertFalse(by_layout["owner_qualified"]["exists"])
            self.assertTrue(by_layout["owner_namespaced"]["exists"])

    def test_owner_qualified_root_is_ambiguous_with_flat_or_namespaced(self) -> None:
        for other_layout in ("flat", "owner_namespaced"):
            with self.subTest(other_layout=other_layout), tempfile.TemporaryDirectory() as temporary:
                input_root = Path(temporary)
                owner_qualified = input_root / "owner" / "qualification-source"
                owner_qualified.mkdir(parents=True)
                self._marker(owner_qualified)
                other = (input_root / "qualification-source" if other_layout == "flat"
                         else input_root / "datasets" / "owner" / "qualification-source")
                other.mkdir(parents=True)
                report = {}

                with self.assertRaisesRegex(ValueError, "exactly one dataset mount"):
                    self.resolver.resolve_dataset_mount(
                        input_root, "owner/qualification-source",
                        marker="modern2d_qualification_launch.json", report=report)

                self.assertEqual(report["status"], "rejected")
                self.assertEqual(report["candidate_count"], 2)
                self.assertEqual(len(report["candidates"]), 3)
                by_layout = {record["layout"]: record for record in report["candidates"]}
                self.assertTrue(by_layout["owner_qualified"]["exists"])
                self.assertTrue(by_layout[other_layout]["exists"])

    def test_no_dataset_root_is_rejected_without_fallback_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = {}
            with self.assertRaisesRegex(ValueError, "exactly one dataset mount"):
                self.resolver.resolve_dataset_mount(
                    Path(temporary), "owner/qualification-source", report=report)
            self.assertEqual(report["status"], "rejected")
            self.assertEqual(report["candidate_count"], 0)
            self.assertEqual(len(report["candidates"]), 3)
            self.assertTrue(all(not item["exists"] for item in report["candidates"]))

    def test_missing_marker_is_rejected_for_all_layouts(self) -> None:
        for layout in ("flat", "owner_qualified", "owner_namespaced"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as temporary:
                input_root = Path(temporary)
                root = {
                    "flat": input_root / "qualification-source",
                    "owner_qualified": input_root / "owner" / "qualification-source",
                    "owner_namespaced": input_root / "datasets" / "owner" / "qualification-source",
                }[layout]
                root.mkdir(parents=True)
                report = {}

                with self.assertRaisesRegex(ValueError, "missing its metadata marker"):
                    self.resolver.resolve_dataset_mount(
                        input_root, "owner/qualification-source",
                        marker="modern2d_qualification_launch.json", report=report)

                self.assertEqual(report["status"], "rejected")
                self.assertEqual(report["candidate_count"], 1)
                self.assertEqual(len(report["candidates"]), 3)

    def test_marker_is_filename_only_and_its_content_is_never_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_root = Path(temporary)
            flat = input_root / "qualification-source"
            flat.mkdir()
            marker = self._marker(flat)
            report = {}

            real_open = Path.open
            real_read_bytes = Path.read_bytes

            def fail_open(path, *args, **kwargs):
                if path == marker:
                    raise AssertionError("mount discovery opened marker content")
                return real_open(path, *args, **kwargs)

            def fail_read_bytes(path, *args, **kwargs):
                if path == marker:
                    raise AssertionError("mount discovery read marker content")
                return real_read_bytes(path, *args, **kwargs)

            with mock.patch.object(Path, "open", fail_open), \
                    mock.patch.object(Path, "read_bytes", fail_read_bytes):
                selected = self.resolver.resolve_dataset_mount(
                    input_root, "owner/qualification-source",
                    marker=marker.name, report=report)

            self.assertEqual(selected, flat)
            self.assertTrue(report["candidates"][0]["marker_present"])

    def test_wrong_owner_slug_mount_name_and_path_traversal_fail_closed(self) -> None:
        invalid = (
            ("owner/qualification-source", "other-source"),
            ("../owner/qualification-source", None),
            ("owner/../qualification-source", None),
            ("owner//qualification-source", None),
            ("owner/qualification-source/extra", None),
            ("owner\\qualification-source", None),
            ("owner/Qualification-source", None),
        )
        with tempfile.TemporaryDirectory() as temporary:
            input_root = Path(temporary)
            namespaced = input_root / "datasets" / "owner" / "qualification-source"
            namespaced.mkdir(parents=True)
            for dataset_source, mount_name in invalid:
                with self.subTest(dataset_source=dataset_source, mount_name=mount_name):
                    report = {}
                    with self.assertRaises(ValueError):
                        self.resolver.resolve_dataset_mount(
                            input_root, dataset_source, mount_name=mount_name, report=report)
                    self.assertEqual(report["status"], "rejected")
                    self.assertIn(len(report["candidates"]), (0, 3))

            for dataset_source in ("other/qualification-source", "owner/other-source"):
                with self.subTest(dataset_source=dataset_source):
                    report = {}
                    with self.assertRaises(ValueError):
                        self.resolver.resolve_dataset_mount(
                            input_root, dataset_source, report=report)
                    self.assertEqual(report["status"], "rejected")
                    self.assertEqual(report["candidate_count"], 0)
                    self.assertEqual(len(report["candidates"]), 3)

    def test_marker_directory_and_marker_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_root = Path(temporary)
            flat = input_root / "qualification-source"
            flat.mkdir()
            (flat / "modern2d_qualification_launch.json").mkdir()
            with self.assertRaisesRegex(ValueError, "metadata marker"):
                self.resolver.resolve_dataset_mount(
                    input_root, "owner/qualification-source",
                    marker="modern2d_qualification_launch.json")

        with tempfile.TemporaryDirectory() as temporary:
            input_root = Path(temporary)
            flat = input_root / "qualification-source"
            target = input_root / "target-marker"
            flat.mkdir()
            target.write_bytes(b"not opened")
            link = flat / "modern2d_qualification_launch.json"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink capability unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "links are forbidden"):
                self.resolver.resolve_dataset_mount(
                    input_root, "owner/qualification-source",
                    marker="modern2d_qualification_launch.json")

    def test_mount_root_symlink_and_windows_junction_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_root = Path(temporary)
            target = input_root / "real-source"
            target.mkdir()
            self._marker(target)
            link = input_root / "qualification-source"
            try:
                link.symlink_to(target, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink capability unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "links are forbidden"):
                self.resolver.resolve_dataset_mount(
                    input_root, "owner/qualification-source",
                    marker="modern2d_qualification_launch.json")

        with tempfile.TemporaryDirectory() as temporary:
            input_root = Path(temporary)
            flat = input_root / "qualification-source"
            flat.mkdir()
            self._marker(flat)
            real_is_junction = getattr(Path, "is_junction", None)

            def fake_is_junction(path):
                return path == flat

            with mock.patch.object(Path, "is_junction", fake_is_junction, create=True):
                with self.assertRaisesRegex(ValueError, "links are forbidden"):
                    self.resolver.resolve_dataset_mount(
                        input_root, "owner/qualification-source",
                        marker="modern2d_qualification_launch.json")
            self.assertTrue(real_is_junction is None or callable(real_is_junction))

    def test_helper_is_stdlib_only_before_any_mount_call(self) -> None:
        real_import = builtins.__import__
        stdlib = set(__import__("sys").stdlib_module_names)

        def stdlib_only(name, *args, **kwargs):
            if name.partition(".")[0] not in stdlib:
                raise AssertionError(f"non-stdlib import in mount resolver: {name}")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", stdlib_only):
            _load_helper()


if __name__ == "__main__":
    unittest.main()
