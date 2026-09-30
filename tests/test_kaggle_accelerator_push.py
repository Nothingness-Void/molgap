from __future__ import annotations

import json
from pathlib import Path

import pytest

from molgap import kaggle_accelerator_push


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"error": None, "invalidDatasetSources": []}


@pytest.mark.parametrize("response_payload", [
    {"error": None, "invalidDatasetSources": []},
    {"error": None, "ref": "owner/actual-title-slug", "versionNumber": 2,
     "url": "https://www.kaggle.com/code/owner/actual-title-slug", "invalidDatasetSources": []},
])
def test_push_kernel_sends_machine_shape_without_exposing_key(
    tmp_path: Path, monkeypatch, response_payload
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
    monkeypatch.setattr(_Response, "json", lambda self: response_payload)
    result = kaggle_accelerator_push.push_kernel_with_accelerator(
        package_dir=package,
        credential_path=credentials,
        accelerator="NvidiaTeslaT4",
    )

    assert result["status"] == "submitted"
    assert captured["json"]["machineShape"] == "NvidiaTeslaT4"
    assert captured["json"]["enableTpu"] is False
    assert captured["json"]["enableGpu"] is True
    assert "secret-value" not in json.dumps(result)
    assert result["kernel"] == response_payload.get("ref")
    assert result["requested_kernel"] == "owner/kernel"
    assert result["reconciliation_required"] is ("ref" not in response_payload)

    def timeout(*args, **kwargs):
        raise kaggle_accelerator_push.requests.Timeout("response lost")
    monkeypatch.setattr(kaggle_accelerator_push.requests, "post", timeout)
    unknown = kaggle_accelerator_push.push_kernel_with_accelerator(package_dir=package,
        credential_path=credentials, accelerator="NvidiaTeslaT4")
    assert unknown["status"] == "submission_unknown"
    assert unknown["kernel"] is None and unknown["reconciliation_required"]
    assert "secret-value" not in json.dumps(unknown)


def test_observed_identity_uses_actual_slug_and_keeps_version():
    result = kaggle_accelerator_push._observed_identity({
        "ref": "owner/title-generated-slug",
        "url": "https://www.kaggle.com/code/owner/title-generated-slug?scriptVersionId=123",
        "versionNumber": 2, "kernelId": 77,
    }, "owner")
    assert result["kernel"] == "owner/title-generated-slug"
    assert result["script_version_id"] == "123"
    assert result["version_number"] == 2
    assert result["kernel_id"] == 77


def test_conflicting_response_identity_is_not_guessed():
    result = kaggle_accelerator_push._observed_identity({
        "ref": "owner/requested-slug", "url": "https://www.kaggle.com/code/owner/other-slug",
    }, "owner")
    assert result["kernel"] is None
    assert result["identity_conflicts"] == ["response_ref_url_conflict"]


def test_push_kernel_rejects_unverified_tpu_batch_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="executed both script and notebook"):
        kaggle_accelerator_push.push_kernel_with_accelerator(
            package_dir=tmp_path,
            credential_path=tmp_path / "kaggle.json",
            accelerator="TpuV5E8",
        )
