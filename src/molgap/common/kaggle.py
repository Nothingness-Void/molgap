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
    seen = set()
    opaque = False
    for page in range(1, 51):
        rows = (api.kernels_list(mine=True, user=owner, page=page, page_size=20)
                if kind == "kernel" else api.dataset_list(mine=True, page=page))
        if not isinstance(rows, list):
            raise RuntimeError("Owned listing is unknown; absence is unverified")
        for row in rows:
            identity = str(row.ref)
            if (kind == "kernel" and not identity and getattr(row, "id", None) == 0
                    and getattr(row, "title", None) == "[Private Notebook]"
                    and getattr(row, "author", None) == ""
                    and getattr(row, "slug", None) == ""):
                opaque = True
                continue
            if identity.split("/")[0].lower() != owner.lower() or identity in seen:
                raise RuntimeError("Owned listing scope/pagination differs")
            if identity.lower() == ref.lower():
                return False
            seen.add(identity)
        if len(rows) < 20:
            if not opaque:
                return True
            # Frozen privacy placeholders have no identity. Do not guess what
            # they hide: ask the authenticated server to filter this exact slug.
            # A matching opaque row, malformed page or duplicate stays UNKNOWN.
            filtered_seen = set()
            for filtered_page in range(1, 51):
                filtered = api.kernels_list(mine=True, user=owner, search=ref.split("/")[1],
                    page=filtered_page, page_size=20)
                if not isinstance(filtered, list):
                    raise RuntimeError("Scoped owned listing is unknown")
                for item in filtered:
                    candidate = str(item.ref)
                    if candidate.lower() == ref.lower():
                        return False
                    if candidate.split("/")[0].lower() != owner.lower() or candidate in filtered_seen:
                        raise RuntimeError("Scoped owned listing is ambiguous")
                    filtered_seen.add(candidate)
                if len(filtered) < 20:
                    return True
            raise RuntimeError("Scoped owned listing exceeded its bounded absence check")
    raise RuntimeError("Owned listing exceeded its bounded absence check")


def dataset_status(api, ref):
    try:
        value = api.dataset_status(ref)
    except Exception as error:
        if is_not_found(error) or owned_absence(api, ref, "dataset", error):
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
        if is_not_found(error) or owned_absence(api, ref, "kernel", error):
            return None
        raise RuntimeError("Kernel identity is unknown; no submission retry") from None
