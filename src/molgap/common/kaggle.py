"""One SDK compatibility boundary for explicit-account, read-only operations."""
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


def platform_error_details(error, *, stage, endpoint_class):
    """Retain bounded transport metadata without request URLs or SDK messages."""
    def label(value, fallback):
        return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value) else fallback

    def status(value):
        return value if type(value) is int and 100 <= value <= 599 else None

    def request_id(value):
        return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9._:/=;+\-]{1,256}", value) else None

    details = {"error_type": label(type(error).__name__, "Exception"),
               "stage": label(stage, "unknown"), "endpoint_class": label(endpoint_class, "other_http")}
    response = getattr(error, "response", None)
    if response is not None:
        url = getattr(response, "url", None)
        if isinstance(url, str):
            try:
                host = (urlsplit(url).hostname or "").lower()
            except ValueError:
                host = ""
            details["endpoint_class"] = ("kaggle_api" if host == "kaggle.com" or host.endswith(".kaggle.com")
                else "google_storage" if host == "storage.googleapis.com" or host.endswith(".storage.googleapis.com")
                else "other_http")
    for value in (getattr(response, "status_code", None), getattr(error, "status", None),
                  getattr(error, "status_code", None), getattr(error, "code", None)):
        if status(value) is not None:
            details["http_status"] = value
            break
    headers = getattr(response, "headers", None)
    if headers is not None:
        for name in ("X-Request-Id", "X-Kaggle-Request-Id", "X-Goog-Request-Id",
                     "X-GUploader-UploadID", "X-Cloud-Trace-Context"):
            value = request_id(headers.get(name))
            if value is None:
                value = request_id(headers.get(name.lower()))
            if value is not None:
                details["request_id"] = value
                break
    if details.get("http_status") == 200:
        content = getattr(response, "content", None)
        if isinstance(content, (bytes, str)) and len(content) <= 16 * 1024:
            try:
                body = json.loads(content)
            except (ValueError, UnicodeError, RecursionError):
                body = None
            if type(body) is dict and status(body.get("code")) is not None:
                details["api_status"] = body["code"]
    retained = getattr(error, "platform_error", None)
    if type(retained) is dict:
        for key in ("error_type", "stage", "endpoint_class"):
            if key in retained:
                details[key] = label(retained[key], details[key])
        for key in ("http_status", "api_status"):
            if status(retained.get(key)) is not None:
                details[key] = retained[key]
        if request_id(retained.get("request_id")) is not None:
            details["request_id"] = retained["request_id"]
    return details


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
    # SDK config loading reads global files/environment and accepts no mapping.
    # Keep an instance-local mapping so neither account nor OAuth can leak in.
    api.config_values = {"username": owner, "key": key}
    return api


def is_not_found(error):
    return (getattr(error, "status", None) == 404 or getattr(error, "status_code", None) == 404
            or getattr(getattr(error, "response", None), "status_code", None) == 404)


def owned_absence(api, ref, kind, error):
    """A private 403 is absence only after a complete authenticated owner listing."""
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status != 403:
        return False
    owner = api.get_config_value(api.CONFIG_NAME_USER)
    if status != 403 or ref.split("/")[0] != owner:
        return False
    return owned_reference_absent(api, ref, kind)


def owned_reference_absent(api, ref, kind):
    """A complete authenticated listing, not a private-page 404 assumption."""
    owner = api.get_config_value(api.CONFIG_NAME_USER)
    if ref.split("/")[0] != owner or kind not in {"kernel", "dataset"}:
        raise ValueError("Absence qualification requires the exact authenticated owner")
    seen = set()
    for page in range(1, 51):
        rows = (api.kernels_list(mine=True, user=owner, page=page, page_size=20)
                if kind == "kernel" else api.dataset_list(mine=True, page=page))
        if not isinstance(rows, list):
            raise RuntimeError("Owned listing is unknown; absence is unverified")
        for row in rows:
            identity = str(row.ref)
            if identity.split("/")[0].lower() != owner.lower() or identity in seen:
                raise RuntimeError("Owned listing scope/pagination differs")
            if identity.lower() == ref.lower():
                return False
            seen.add(identity)
        if len(rows) < 20:
            return True
    raise RuntimeError("Owned listing exceeded its bounded absence check")


def dataset_status(api, ref):
    try:
        value = api.dataset_status(ref)
    except Exception as error:
        if is_not_found(error) or owned_absence(api, ref, "dataset", error):
            return None
        wrapped = RuntimeError("Dataset status is unknown; no publication retry")
        wrapped.platform_error = platform_error_details(error, stage="dataset_status", endpoint_class="kaggle_api")
        raise wrapped from None
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


def dataset_file_url(api, ref, name, version):
    """Resolve one raw, version-pinned file through the authenticated SDK."""
    from urllib.parse import urlsplit
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest
    request = ApiDownloadDatasetRequest()
    request.owner_slug, request.dataset_slug = ref.split("/")
    request.file_name, request.dataset_version_number, request.raw = name, version, True
    with api.build_kaggle_client() as client:
        response = client.datasets.dataset_api_client.download_dataset(request)
        try:
            url = response.url
            parsed = urlsplit(url)
            if response.status_code != 200 or parsed.scheme != "https" or not parsed.hostname:
                raise RuntimeError("Pinned source file redirect is unavailable")
            return url
        finally:
            response.close()


def kernel_source(api, ref):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = ref.split("/")
    try:
        with api.build_kaggle_client() as client:
            return client.kernels.kernels_api_client.get_kernel(request)
    except Exception as error:
        if is_not_found(error) or owned_absence(api, ref, "kernel", error):
            return None
        wrapped = RuntimeError("Kernel identity is unknown; no submission retry")
        wrapped.platform_error = platform_error_details(error, stage="kernel_source", endpoint_class="kaggle_api")
        raise wrapped from None
