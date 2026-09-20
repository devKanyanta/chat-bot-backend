"""In-memory per-IP fixed-window rate limiter (spec section 11)."""

import threading
import time
from collections import defaultdict, deque

from fastapi import Request

from .config import get_settings
from .errors import RateLimitedError


class RateLimiter:
    """Fixed-window counter per client IP. In-memory; resets on restart."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, request: Request) -> None:
        limit = get_settings().rate_limit_per_minute
        if limit <= 0:
            return

        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window_start = now - 60.0

        with self._lock:
            hits = self._hits[ip]
            while hits and hits[0] <= window_start:
                hits.popleft()
            if len(hits) >= limit:
                raise RateLimitedError()
            hits.append(now)


# Shared instance for the app's lifetime.
limiter = RateLimiter()
