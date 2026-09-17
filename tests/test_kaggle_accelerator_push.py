from __future__ import annotations

import json
from pathlib import Path

from molgap import kaggle_accelerator_push


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"error": None, "invalidDatasetSources": []}


def test_push_kernel_sends_machine_shape_without_exposing_key(
    tmp_path: Path, monkeypatch
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "script.py").write_text("print('ok')\n", encoding="utf-8")
    (package / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "owner/kernel",
                "title": "Kernel title",
                "code_file": "script.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_internet": "false",
                "dataset_sources": ["owner/data"],
            }
        ),
        encoding="utf-8",
    )
    credentials = tmp_path / "kaggle.json"
    credentials.write_text(
        json.dumps({"username": "owner", "key": "secret-value"}),
        encoding="utf-8",
    )
    captured = {}

    def fake_post(url, *, auth, json, timeout):
        captured.update({"url": url, "auth": auth, "json": json, "timeout": timeout})
        return _Response()

    monkeypatch.setattr(kaggle_accelerator_push.requests, "post", fake_post)
    result = kaggle_accelerator_push.push_kernel_with_accelerator(
        package_dir=package,
        credential_path=credentials,
        accelerator="TpuV5E8",
    )

    assert result["status"] == "submitted"
    assert captured["json"]["machineShape"] == "TpuV5E8"
    assert captured["json"]["enableTpu"] is True
    assert captured["json"]["enableGpu"] is False
    assert "secret-value" not in json.dumps(result)
