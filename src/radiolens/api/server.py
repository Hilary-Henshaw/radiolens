"""FastAPI application factory and entrypoint."""

from __future__ import annotations

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from radiolens.api.endpoints import router
from radiolens.api.guards import RequestAuditMiddleware, UploadSizeGuard
from radiolens.api.providers import lifespan
from radiolens.config import get_settings

log = structlog.get_logger(__name__)


def create_api_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Registers:

    - ``lifespan`` context manager for model loading / unloading.
    - :class:`~fastapi.middleware.cors.CORSMiddleware` using
      ``settings.cors_allow_origins``.
    - :class:`~radiolens.api.guards.UploadSizeGuard` enforcing
      ``settings.max_upload_bytes``.
    - :class:`~radiolens.api.guards.RequestAuditMiddleware` for structured
      per-request logging.
    - API router mounted at ``/api/v1``.
    - Root redirect from ``/`` to ``/docs``.

    Returns:
        Fully configured :class:`fastapi.FastAPI` instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="radiolens",
        description=(
            "Clinical-grade chest X-ray pneumonia detection API. "
            "RESEARCH USE ONLY — not intended for clinical decision-making."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(
        UploadSizeGuard,
        max_bytes=settings.max_upload_bytes,
    )

    app.add_middleware(RequestAuditMiddleware)

    app.include_router(router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def _root_redirect() -> RedirectResponse:
        """Redirect root URL to the interactive API documentation."""
        return RedirectResponse(url="/docs")

    log.info(
        "app_created",
        cors_origins=settings.cors_allow_origins,
        max_upload_bytes=settings.max_upload_bytes,
    )
    return app


def serve() -> None:
    """Start the uvicorn server using environment-sourced settings.

    Reads ``api_host`` and ``api_port`` from the
    :class:`~radiolens.config.Settings` singleton.  Structured logging is
    handled entirely by ``structlog``; the uvicorn default log config is
    disabled.
    """
    settings = get_settings()
    log.info(
        "server_starting",
        host=settings.api_host,
        port=settings.api_port,
    )
    uvicorn.run(
        "radiolens.api.server:create_api_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,
    )


app = create_api_app()
