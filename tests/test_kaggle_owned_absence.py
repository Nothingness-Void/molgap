from types import SimpleNamespace

import pytest

from molgap.common.kaggle import owned_absence


class Api:
    CONFIG_NAME_USER = "username"

    def __init__(self, filtered):
        self.filtered = filtered
        self.calls = []

    def get_config_value(self, key):
        return "owner"

    def kernels_list(self, **kwargs):
        self.calls.append(kwargs)
        if "search" in kwargs:
            return self.filtered
        return [SimpleNamespace(ref="owner/existing"), SimpleNamespace(ref="", id=0,
            title="[Private Notebook]", author="", slug="")]


def forbidden():
    return SimpleNamespace(response=SimpleNamespace(status_code=403))


def test_frozen_private_placeholders_require_scoped_authoritative_listing():
    api = Api([])
    assert owned_absence(api, "owner/new-question", "kernel", forbidden())
    assert api.calls[-1]["search"] == "new-question"


def test_scoped_match_is_not_absent():
    assert not owned_absence(Api([SimpleNamespace(ref="owner/new-question")]),
                             "owner/new-question", "kernel", forbidden())


@pytest.mark.parametrize("rows", [None, [SimpleNamespace(ref="")],
    [SimpleNamespace(ref="someone/other")],
    [SimpleNamespace(ref="owner/other"), SimpleNamespace(ref="owner/other")]])
def test_scoped_unknown_rows_stay_fail_closed(rows):
    with pytest.raises(RuntimeError):
        owned_absence(Api(rows), "owner/new-question", "kernel", forbidden())


def test_unrecognized_blank_is_not_a_privacy_placeholder():
    api = Api([])
    api.kernels_list = lambda **kwargs: [SimpleNamespace(ref="")]
    with pytest.raises(RuntimeError):
        owned_absence(api, "owner/new-question", "kernel", forbidden())


def test_foreign_owner_never_qualifies_absence():
    api = Api([])
    assert not owned_absence(api, "foreign/new-question", "kernel", forbidden())
    assert not api.calls
