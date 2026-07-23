"""
TTL cache with a transparent in-memory fallback.

Ported from the NYC Transit Hub backend (backend/services/cache.py): callers use
one get/set API and never learn whether Redis or the in-process dict is active,
so the alerts server keeps serving cached data through a Redis outage.
"""
from __future__ import annotations

import json
import time

try:  # redis is optional — absence just forces the in-memory path
    import redis  # type: ignore
except Exception:  # pragma: no cover - exercised only when redis is missing
    redis = None  # type: ignore


class TTLCache:
    def __init__(self, redis_url: str | None = None):
        self._client = None
        self._use_memory = True
        self._mem: dict[str, tuple[object, float | None]] = {}
        if redis_url and redis is not None:
            try:
                self._client = redis.Redis.from_url(redis_url, socket_connect_timeout=2)
                self._client.ping()
                self._use_memory = False
            except Exception:
                self._use_memory = True

    @property
    def backend(self) -> str:
        return "memory" if self._use_memory else "redis"

    def get(self, key: str):
        if self._use_memory:
            entry = self._mem.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at is not None and time.time() > expires_at:
                self._mem.pop(key, None)
                return None
            return value
        raw = self._client.get(key)
        return None if raw is None else json.loads(raw)

    def set(self, key: str, value, ttl: int = 60) -> None:
        if self._use_memory:
            self._mem[key] = (value, time.time() + ttl if ttl else None)
            return
        self._client.setex(key, ttl, json.dumps(value))

    def delete(self, key: str) -> None:
        if self._use_memory:
            self._mem.pop(key, None)
            return
        self._client.delete(key)
