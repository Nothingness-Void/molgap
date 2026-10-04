"""Use the requested API key even when the SDK finds another OAuth session."""
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
