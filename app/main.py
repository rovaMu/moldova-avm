"""FastAPI core setup: app factory, CORS middleware and a global exception
handler so the mobile client always receives a structured JSON error.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, get_settings
from app.routes import formula

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("moldova_avm")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "Automated Valuation Model backend for the Moldovan property "
            "market. Crawls five sources, normalizes multilingual features "
            "and exposes dual regression formulas to mobile clients."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(formula.router, prefix=settings.api_prefix)

    # --- Global exception handlers --------------------------------------
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "error": {"code": exc.status_code, "message": exc.detail},
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "error": {"code": 422, "message": "Validation error",
                          "details": exc.errors()},
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": {"code": 500, "message": "Internal server error"},
            },
        )

    @app.get("/", tags=["health"], summary="Service metadata")
    async def root() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name, "version": "1.0.0"}

    @app.get("/health", tags=["health"], summary="Health check")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    return app


app = create_app()
