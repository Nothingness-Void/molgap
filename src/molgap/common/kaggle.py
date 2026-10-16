"""One SDK compatibility boundary for explicit-account, read-only operations."""
import json
from pathlib import Path


def api_for_credentials(path: Path):
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk import KaggleClient

    credentials = json.loads(Path(path).read_text(encoding="utf-8"))
    owner, key = credentials["username"], credentials["key"]

    class ExplicitAccountApi(KaggleApi):
        def build_kaggle_client(self):
            client = KaggleClient(username=owner, password=key)
            # The installed SDK prefers a global OAuth token over these arguments.
            # Bind Basic auth on this short-lived client, leaving login files intact.
            def bind_auth():
                client._http_client._session.auth = (owner, key)
                client._http_client._signed_in = True
            client._http_client._try_fill_auth = bind_auth
            return client

    api = ExplicitAccountApi()
    api._load_config({"username": owner, "key": key})
    return api


def is_not_found(error):
    code = (getattr(error, "status", None) or getattr(error, "status_code", None)
            or getattr(getattr(error, "response", None), "status_code", None))
    if code == 404:
        return True
    if code == 403:
        text = getattr(getattr(error, "response", None), "text", "") or ""
        return "PERMISSION_DENIED" in text and (
            "kernels.get" in text or "datasets.get" in text)
    return False


def dataset_status(api, ref):
    try:
        value = api.dataset_status(ref)
    except Exception as error:
        if is_not_found(error):
            return None
        raise RuntimeError("Dataset status is unknown; no publication retry") from None
    if type(value) is not str or not value:
        raise ValueError("Malformed dataset status; no publication retry")
    return {"status": value.lower()}


def dataset_details(api, ref):
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    request = ApiGetDatasetRequest()
    request.owner_slug, request.dataset_slug = ref.split("/")
    with api.build_kaggle_client() as client:
        details = client.datasets.dataset_api_client.get_dataset(request)
    if details.ref != ref or details.is_private is not True:
        raise ValueError("Remote source dataset identity/private binding differs")
    return details


def kernel_source(api, ref):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = ref.split("/")
    try:
        with api.build_kaggle_client() as client:
            return client.kernels.kernels_api_client.get_kernel(request)
    except Exception as error:
        if is_not_found(error):
            return None
        raise RuntimeError("Kernel identity is unknown; no submission retry") from None
