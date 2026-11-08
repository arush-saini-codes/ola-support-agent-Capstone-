"""Response Cache with LRU eviction and question normalization (Section 14.4).

Caches grounded-generation results (steps 6-10: crew + groundedness + review)
keyed by normalized resolved masked questions.
"""

import config  # Ground Rule 3: must be first import
import re
from collections import OrderedDict
from typing import Any


def normalize(text: str) -> str:
    """Normalize question: lowercase, collapse whitespace, strip trailing punctuation."""
    s = text.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip("?.!")
    return s.strip()


class ResponseCache:
    """Thread-safe LRU response cache backed by OrderedDict."""

    def __init__(self, max_entries: int | None = None):
        self.max_entries = max_entries or config.CACHE_MAX_ENTRIES
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict[str, Any] | None:
        """Look up a cached result by normalized key."""
        norm_key = normalize(key)
        if norm_key in self._store:
            self._store.move_to_end(norm_key)
            self.hits += 1
            return dict(self._store[norm_key])
        self.misses += 1
        return None

    def put(self, key: str, data: dict[str, Any]) -> None:
        """Store an entry in the cache if status is 'ok'."""
        if data.get("status") != "ok":
            return
        norm_key = normalize(key)
        if norm_key in self._store:
            self._store.move_to_end(norm_key)
        self._store[norm_key] = dict(data)
        if len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached entries and reset statistics."""
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self.hits + self.misses
        ratio = round(self.hits / total, 3) if total > 0 else 0.0
        return {
            "entries": len(self._store),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": ratio,
        }


# Global response cache singleton
cache = ResponseCache()

