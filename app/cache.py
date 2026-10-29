from __future__ import annotations

from threading import RLock
from typing import Any

from cachetools import TTLCache


class MemoryCache:
    """Thread-safe wrapper around cachetools.TTLCache."""

    def __init__(self, ttl: int = 600, maxsize: int = 10_000):
        if ttl < 1:
            raise ValueError("ttl must be at least 1 second")
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")

        self._ttl = ttl
        self._maxsize = maxsize
        self._lock = RLock()
        self._cache: TTLCache[str, Any] = TTLCache(maxsize=maxsize, ttl=ttl)

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = value

    def flush(self) -> None:
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def change_ttl(self, ttl: int) -> None:
        if ttl < 1:
            raise ValueError("ttl must be at least 1 second")

        with self._lock:
            self._ttl = ttl
            self._cache = TTLCache(maxsize=self._maxsize, ttl=ttl)

    @property
    def ttl(self) -> int:
        return self._ttl