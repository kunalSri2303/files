import hashlib
import time
from collections import OrderedDict
from typing import Optional, Any
from utils.logger import app_logger


class TTLCache:
    """
    Thread-safe in-memory LRU cache with TTL expiry.
    Uses an OrderedDict to maintain insertion order for LRU eviction.
    """

    def __init__(self, max_size: int = 500, ttl: int = 3600):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(text: str, length: str) -> str:
        """Create a deterministic cache key from text + length."""
        raw = f"{length}::{text}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            self._misses += 1
            return None

        value, expiry = self._cache[key]

        if time.monotonic() > expiry:
            del self._cache[key]
            self._misses += 1
            app_logger.debug("Cache expired", extra={"key": key[:16]})
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        app_logger.debug("Cache hit", extra={"key": key[:16]})
        return value

    def set(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                evicted, _ = self._cache.popitem(last=False)
                app_logger.debug("Cache evicted LRU entry", extra={"key": evicted[:16]})
        self._cache[key] = (value, time.monotonic() + self.ttl)

    def stats(self) -> dict:
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (
                round(self._hits / (self._hits + self._misses), 3)
                if (self._hits + self._misses) > 0
                else 0.0
            ),
        }


# Singleton cache instance used by the service layer
summarizer_cache = TTLCache()
