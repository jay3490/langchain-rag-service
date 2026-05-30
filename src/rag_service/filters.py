"""Backend-neutral metadata filters applied at retrieval time.

A :class:`MetadataFilter` is a conjunction of conditions. Each condition names a
metadata key and the values it may take: ``{"source": "bm25.md"}`` keeps chunks of one
file, ``{"source": ["bm25.md", "mmr.md"]}`` keeps chunks of either. Every backend
translates the same filter: BM25 and FAISS evaluate it in Python with :meth:`matches`,
Qdrant receives an equivalent payload filter.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

Scalar = str | int | bool
FilterValue = Scalar | list[Scalar]

#: Metadata key that holds the owning tenant; it is set by the index, never by clients.
TENANT_KEY = "tenant_id"


def _as_values(value: object) -> tuple[Scalar, ...]:
    values = list(value) if isinstance(value, list | tuple | set | frozenset) else [value]
    if not values:
        raise ValueError("a filter condition needs at least one value")
    for item in values:
        if not isinstance(item, str | int | bool):
            raise ValueError(f"filter values must be strings, integers or booleans, got {item!r}")
    return tuple(values)


@dataclass(frozen=True)
class MetadataFilter:
    """Chunks match when, for every condition, their metadata value is one of the values."""

    conditions: tuple[tuple[str, tuple[Scalar, ...]], ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> MetadataFilter:
        if not mapping:
            return cls()
        return cls(tuple(sorted((str(key), _as_values(value)) for key, value in mapping.items())))

    @property
    def is_empty(self) -> bool:
        return not self.conditions

    def keys(self) -> Iterable[str]:
        return (key for key, _ in self.conditions)

    def with_condition(self, key: str, value: Scalar) -> MetadataFilter:
        """Return a copy where ``key`` must equal ``value``, replacing any condition on it."""
        kept = [(k, v) for k, v in self.conditions if k != key]
        return MetadataFilter(tuple(sorted([*kept, (key, (value,))])))

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        for key, values in self.conditions:
            if key not in metadata:
                return False
            actual = metadata[key]
            # bool is an int subclass; keep True from matching 1 and vice versa.
            if not any(type(actual) is type(v) and actual == v for v in values):
                return False
        return True

    def as_dict(self) -> dict[str, list[Scalar]]:
        """Canonical, JSON-serialisable form (used in cache keys and responses)."""
        return {key: list(values) for key, values in self.conditions}
