from pathlib import Path

import pytest

from molgap.presentation_figures import (
    DARK_THEME,
    LIGHT_THEME,
    _resolve_theme,
    build_manifest,
)


def test_theme_names_resolve_case_insensitively() -> None:
    assert _resolve_theme("light") is LIGHT_THEME
    assert _resolve_theme("DARK") is DARK_THEME


def test_unknown_theme_fails_closed() -> None:
    with pytest.raises(ValueError, match="light.*dark"):
        _resolve_theme("neon")


def test_manifest_excludes_inspection_renders(tmp_path: Path) -> None:
    (tmp_path / "07_track_a_accuracy.png").write_bytes(b"png")
    (tmp_path / "02_routes_abc_inspect.png").write_bytes(b"png")
    build_manifest(tmp_path, LIGHT_THEME)
    manifest = (tmp_path / "figure_manifest.json").read_text(encoding="utf-8")
    assert "07_track_a_accuracy.png" in manifest
    assert "02_routes_abc_inspect.png" not in manifest
