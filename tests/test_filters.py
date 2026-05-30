import pytest

from rag_service.filters import MetadataFilter


def test_empty_filter_matches_everything() -> None:
    assert MetadataFilter.from_mapping(None).is_empty
    assert MetadataFilter.from_mapping({}).matches({"source": "a.md"})


def test_scalar_and_list_conditions() -> None:
    metadata_filter = MetadataFilter.from_mapping({"source": ["a.md", "b.md"], "page": 2})

    assert metadata_filter.matches({"source": "b.md", "page": 2})
    assert not metadata_filter.matches({"source": "c.md", "page": 2})
    assert not metadata_filter.matches({"source": "a.md", "page": 3})
    assert not metadata_filter.matches({"source": "a.md"})


def test_booleans_and_integers_do_not_match_each_other() -> None:
    assert not MetadataFilter.from_mapping({"draft": True}).matches({"draft": 1})
    assert not MetadataFilter.from_mapping({"page": 1}).matches({"page": True})
    assert MetadataFilter.from_mapping({"draft": False}).matches({"draft": False})


def test_with_condition_replaces_an_existing_condition() -> None:
    metadata_filter = MetadataFilter.from_mapping({"tenant_id": "evil", "source": "a.md"})
    enforced = metadata_filter.with_condition("tenant_id", "acme")

    assert enforced.as_dict() == {"source": ["a.md"], "tenant_id": ["acme"]}
    assert not enforced.matches({"tenant_id": "evil", "source": "a.md"})


def test_filters_are_canonical() -> None:
    first = MetadataFilter.from_mapping({"b": 1, "a": "x"})
    second = MetadataFilter.from_mapping({"a": ["x"], "b": [1]})
    assert first == second


@pytest.mark.parametrize("value", [[], 1.5, None, {"nested": 1}])
def test_invalid_values_are_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        MetadataFilter.from_mapping({"key": value})
