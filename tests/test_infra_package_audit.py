"""Focused offline regressions for source-package dependency boundaries."""

import pytest

from molgap.source_dependencies import discover_source_dependencies


def _write(root, name, source):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_data_payload_module_is_not_admitted_as_source(tmp_path):
    _write(tmp_path, "src/molgap/__init__.py", "")
    _write(tmp_path, "src/molgap/data/payload.py", "VALUE = 1\n")

    with pytest.raises(ValueError):
        discover_source_dependencies(tmp_path, ["src/molgap/data/payload.py"])


def test_reviewed_io_modules_and_payload_named_source_are_admitted(tmp_path):
    _write(tmp_path, "src/molgap/__init__.py", "")
    _write(tmp_path, "src/molgap/data/fixed_records.py", "VALUE = 1\n")
    _write(tmp_path, "src/molgap/conservative_fusion_payload.py", "VALUE = 2\n")

    closure = discover_source_dependencies(tmp_path, [
        "src/molgap/data/fixed_records.py",
        "src/molgap/conservative_fusion_payload.py",
    ])

    assert "src/molgap/data/fixed_records.py" in closure.source_files
    assert "src/molgap/conservative_fusion_payload.py" in closure.source_files


@pytest.mark.parametrize("import_statement, expression", [
    ("import molgap\n", "molgap.Symbol"),
    ("import molgap as m\n", "m.Symbol"),
])
def test_plain_package_import_traces_lazy_attribute_module(
        tmp_path, import_statement, expression):
    _write(tmp_path, "src/molgap/__init__.py", (
        "def __getattr__(name):\n"
        "    from .public import Symbol\n"
        "    return Symbol\n"
    ))
    _write(tmp_path, "src/molgap/plugin.py", (
        import_statement +
        "def run():\n" +
        f"    return {expression}\n"
    ))
    _write(tmp_path, "src/molgap/public.py", "class Symbol: pass\n")

    closure = discover_source_dependencies(
        tmp_path, ["src/molgap/plugin.py"], require_static=True)

    assert "src/molgap/public.py" in closure.source_files
