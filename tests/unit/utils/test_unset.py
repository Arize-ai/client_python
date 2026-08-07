"""Unit tests for the shared PATCH unset sentinel."""

from arize.utils.unset import _UNSET, UNSET, is_provided


def test_unset_is_not_provided() -> None:
    """The shared sentinel represents an omitted PATCH argument."""
    assert isinstance(_UNSET, UNSET)
    assert not is_provided(_UNSET)


def test_explicit_none_is_provided() -> None:
    """None remains distinguishable from an omitted PATCH argument."""
    assert is_provided(None)
