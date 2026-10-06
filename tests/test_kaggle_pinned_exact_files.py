"""Offline exact-file transport tests; no Kaggle account, graph or model."""
import hashlib
from types import SimpleNamespace
import pytest
from molgap.kaggle_output_retrieval import retrieve_pinned_exact_files


class Api:
    CONFIG_NAME_USER = "user"
    def get_config_value(self, _):
        return "owner"
    def build_kaggle_client(self):
        return self
    def __enter__(self):
        self.kernels = SimpleNamespace(kernels_api_client=self)
        return self
    def __exit__(self, *_):
        pass
    def download_kernel_output(self, request):
        self.request = request
        return SimpleNamespace(url="https://signed.invalid/secret")
    def list_kernel_session_output(self, _):
        raise AssertionError("Exact retrieval must not list the worker environment")


class Http:
    status_code = 200
    def get(self, url, **_):
        assert url == "https://signed.invalid/secret"
        return self
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def iter_content(self, _):
        yield b"retained prediction bytes"


def context(version=3):
    return SimpleNamespace(account="owner",run_reference="owner/kernel",platform_version=version)


def test_exact_binary_version_hash_and_idempotency(tmp_path):
    api = Api()
    pin = hashlib.sha256(b"retained prediction bytes").hexdigest()
    files = {"run/predictions.pt": ("run/predictions.pt",pin)}
    args = dict(context=context(),remote_manifest_path="run/manifest.json",files=files,
                destination=tmp_path,http_session=Http())
    result = retrieve_pinned_exact_files(api,**args)
    assert api.request.version_number == 3
    assert api.request.file_path == "run/predictions.pt"
    assert result["files"][0]["sha256"] == pin
    assert result["platform_version_verified_by_output_api"] is True
    assert "secret" not in str(result)
    api.download_kernel_output = lambda _: pytest.fail("Retained file must not be requested again")
    assert retrieve_pinned_exact_files(api,**args)["files"][0]["state"] == "ALREADY_RETAINED"


@pytest.mark.parametrize("version",[0,"3",True])
def test_physical_version_required(tmp_path,version):
    with pytest.raises(ValueError,match="physical version"):
        retrieve_pinned_exact_files(Api(),context=context(version),remote_manifest_path="manifest.json",
            files={"p.pt":("p.pt","a"*64)},destination=tmp_path)


def test_hash_mismatch_and_path_escape_fail_closed(tmp_path):
    with pytest.raises(ValueError,match="hash disagrees"):
        retrieve_pinned_exact_files(Api(),context=context(),remote_manifest_path="manifest.json",
            files={"p.pt":("p.pt","a"*64)},destination=tmp_path,http_session=Http())
    assert not (tmp_path/"p.pt").exists()
    with pytest.raises(ValueError):
        retrieve_pinned_exact_files(Api(),context=context(),remote_manifest_path="manifest.json",
            files={"p.pt":("../p.pt","a"*64)},destination=tmp_path,http_session=Http())
