"""ASGI middleware: file size enforcement and structured request logging."""

from __future__ import annotations

import time

import structlog
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

log = structlog.get_logger(__name__)

_HTTP_413: int = 413


class UploadSizeGuard(BaseHTTPMiddleware):
    """Reject multipart uploads whose Content-Length exceeds the limit.

    The check is performed on the ``Content-Length`` request header before
    the body is read.  If the header is absent the request is passed through
    (the application layer is responsible for streaming limits).

    Args:
        app: The downstream ASGI application.
        max_bytes: Maximum allowed body size in bytes.

    Example:
        >>> app.add_middleware(UploadSizeGuard, max_bytes=10 * 1024 * 1024)
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Return HTTP 413 if Content-Length header exceeds limit.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            A :class:`JSONResponse` with status 413 if the upload is too
            large, otherwise the downstream response.
        """
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                size = int(content_length)
            except ValueError:
                size = 0

            if size > self._max_bytes:
                log.warning(
                    "upload_rejected_too_large",
                    content_length=size,
                    max_bytes=self._max_bytes,
                    path=request.url.path,
                )
                return JSONResponse(
                    status_code=_HTTP_413,
                    content={
                        "error": "payload_too_large",
                        "detail": (
                            f"Upload size {size} bytes exceeds the "
                            f"{self._max_bytes}-byte limit."
                        ),
                        "status_code": _HTTP_413,
                    },
                )

        return await call_next(request)


class RequestAuditMiddleware(BaseHTTPMiddleware):
    """Log method, path, status code, and latency for every HTTP request.

    Emits a structured log at INFO level after the response is sent,
    including ``duration_ms`` (wall-clock milliseconds).

    Example log event::

        {
            "event": "request_handled",
            "method": "POST",
            "path": "/api/v1/classify",
            "status_code": 200,
            "duration_ms": 123.4
        }
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Emit a structured log entry after the response is produced.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            The downstream response, unchanged.
        """
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000.0

        log.info(
            "request_handled",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response
