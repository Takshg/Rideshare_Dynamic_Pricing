from __future__ import annotations

import logging

from fastapi import FastAPI

from dynamic_pricing.api.middleware import logging_and_metrics_middleware
from dynamic_pricing.api.routes import router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="Dynamic Pricing API",
    version="1.1.0",
    description="API for serving dynamic ride pricing predictions and explanations.",
)

app.middleware("http")(logging_and_metrics_middleware)

app.include_router(router)