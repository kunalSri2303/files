import time
from collections import defaultdict, deque
from fastapi import Request, HTTPException, status
from utils.config import get_settings
from utils.logger import app_logger

settings = get_settings()


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter keyed by client IP.
    Tracks request timestamps within a rolling time window.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._windows: dict[str, deque] = defaultdict(deque)

    def is_allowed(self, client_id: str) -> tuple[bool, dict]:
        now = time.monotonic()
        window = self._windows[client_id]

        # Drop timestamps outside the current window
        while window and window[0] <= now - self.window_seconds:
            window.popleft()

        remaining = max(0, self.max_requests - len(window))
        reset_at = int(time.time() + self.window_seconds)

        if len(window) >= self.max_requests:
            return False, {
                "X-RateLimit-Limit": str(self.max_requests),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_at),
                "Retry-After": str(self.window_seconds),
            }

        window.append(now)
        return True, {
            "X-RateLimit-Limit": str(self.max_requests),
            "X-RateLimit-Remaining": str(remaining - 1),
            "X-RateLimit-Reset": str(reset_at),
        }


_limiter = SlidingWindowRateLimiter(
    max_requests=settings.RATE_LIMIT_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW,
)


async def rate_limit_dependency(request: Request):
    """FastAPI dependency that enforces rate limiting per IP."""
    client_ip = request.client.host if request.client else "unknown"
    allowed, headers = _limiter.is_allowed(client_ip)

    # Always attach rate limit headers to the response
    for header, value in headers.items():
        request.state.rate_limit_headers = headers

    if not allowed:
        app_logger.warning(
            "Rate limit exceeded",
            extra={"client_ip": client_ip, "limit": settings.RATE_LIMIT_REQUESTS},
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "status": "error",
                "error": "Rate limit exceeded",
                "detail": (
                    f"Maximum {settings.RATE_LIMIT_REQUESTS} requests per "
                    f"{settings.RATE_LIMIT_WINDOW} seconds allowed."
                ),
                "code": 429,
            },
            headers=headers,
        )
