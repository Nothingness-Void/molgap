"""Synthetic Kaggle owner-list absence checks; no API or network access."""
from types import SimpleNamespace as NS

import pytest

from molgap.common.kaggle import owned_absence


class Forbidden(Exception):
    def __init__(self, status=403):
        self.response = NS(status_code=status)


class FakeApi:
    CONFIG_NAME_USER = "username"

    def __init__(self, pages=(), *, owner="owner"):
        self.pages = pages
        self.owner = owner
        self.list_calls = []
        self.config_calls = []

    def get_config_value(self, name):
        self.config_calls.append(name)
        assert name == self.CONFIG_NAME_USER
        return self.owner

    def kernels_list(self, **kwargs):
        self.list_calls.append(kwargs)
        return self.pages[kwargs["page"] - 1]


def expected_kwargs(page, slug="target_slug"):
    return {
        "mine": True,
        "user": "owner",
        "search": slug,
        "page": page,
        "page_size": 20,
    }


def test_private_target_found_uses_exact_authenticated_slug_search():
    api = FakeApi([[NS(ref="OWNER/TARGET_SLUG")]])

    assert owned_absence(api, "owner/target_slug", "kernel", Forbidden()) is False
    assert api.list_calls == [expected_kwargs(1)]


def test_complete_empty_target_search_proves_absence():
    api = FakeApi([[]])

    assert owned_absence(api, "owner/target_slug", "kernel", Forbidden()) is True
    assert api.list_calls == [expected_kwargs(1)]


def test_unrelated_legitimate_owner_row_does_not_count_as_target():
    api = FakeApi([[NS(ref="owner/another_notebook")]])

    assert owned_absence(api, "owner/target_slug", "kernel", Forbidden()) is True
    assert api.list_calls == [expected_kwargs(1)]


@pytest.mark.parametrize(("status", "ref"), [
    (500, "owner/target_slug"),
    (403, "another_owner/target_slug"),
])
def test_non403_or_wrong_owner_short_circuits_before_listing(status, ref):
    api = FakeApi([[]])

    assert owned_absence(api, ref, "kernel", Forbidden(status)) is False
    assert api.list_calls == []


@pytest.mark.parametrize("bad_rows", [
    [NS(ref="")],
    [NS(ref="another_owner/private_notebook")],
    [NS(ref="owner/unrelated"), NS(ref="owner/unrelated")],
])
def test_invalid_owner_listing_rows_fail_closed(bad_rows):
    api = FakeApi([bad_rows])

    with pytest.raises(RuntimeError, match="Owned listing scope/pagination differs"):
        owned_absence(api, "owner/target_slug", "kernel", Forbidden())


def test_non_list_owner_listing_result_fails_closed():
    api = FakeApi([None])

    with pytest.raises(RuntimeError, match="Owned listing is unknown"):
        owned_absence(api, "owner/target_slug", "kernel", Forbidden())


@pytest.mark.parametrize("second_page", [
    [NS(ref="owner/target_slug")],
    [],
])
def test_full_page_continues_to_target_or_complete_empty_second_page(second_page):
    first_page = [NS(ref=f"owner/unrelated_{index:02d}") for index in range(20)]
    api = FakeApi([first_page, second_page])

    result = owned_absence(api, "owner/target_slug", "kernel", Forbidden())

    assert result is (second_page == [])
    assert api.list_calls == [expected_kwargs(1), expected_kwargs(2)]
