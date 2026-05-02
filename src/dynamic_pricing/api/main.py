from __future__ import annotations

from fastapi import FastAPI

from dynamic_pricing.api.routes import router


app = FastAPI(
    title="Dynamic Pricing API",
    version="1.0.0",
    description="API for serving dynamic ride pricing predictions.",
)

app.include_router(router)