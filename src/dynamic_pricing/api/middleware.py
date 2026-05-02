from __future__ import annotations

import logging
import time

from fastapi import Request
from prometheus_client import Counter, Histogram


logger = logging.getLogger("dynamic_pricing_api")


REQUEST_COUNT = Counter(
    "dynamic_pricing_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "dynamic_pricing_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)


async def logging_and_metrics_middleware(request: Request, call_next):
    start = time.time()

    response = await call_next(request)

    duration = time.time() - start
    endpoint = request.url.path
    method = request.method
    status_code = str(response.status_code)

    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

    logger.info(
        "request_completed method=%s endpoint=%s status=%s duration=%.4f",
        method,
        endpoint,
        status_code,
        duration,
    )

    return response