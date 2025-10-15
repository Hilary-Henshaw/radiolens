"""FastAPI dependency providers (DI container)."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from radiolens.config import Settings, get_settings
from radiolens.core.detector import ThoraxClassifier

log = structlog.get_logger(__name__)

_classifier_instance: ThoraxClassifier | None = None
_startup_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the classifier on startup and release it on shutdown.

    This function is intended to be passed as the ``lifespan`` argument
    to :class:`fastapi.FastAPI`.  It sets the module-level
    ``_classifier_instance`` singleton and records the startup timestamp.

    Args:
        app: The FastAPI application instance (unused directly).

    Yields:
        Nothing — control is yielded back to FastAPI while the server runs.
    """
    global _classifier_instance, _startup_time

    settings = get_settings()
    _classifier_instance = ThoraxClassifier(settings)
    _classifier_instance.load_weights(settings.model_weights_path)
    _startup_time = time.monotonic()

    log.info(
        "classifier_ready",
        path=str(settings.model_weights_path),
    )

    yield

    _classifier_instance = None
    log.info("classifier_released")


def provide_classifier() -> ThoraxClassifier:
    """FastAPI dependency: return the loaded :class:`ThoraxClassifier`.

    Returns:
        The singleton :class:`ThoraxClassifier` instance.

    Raises:
        RuntimeError: If the classifier has not been initialised via
            ``lifespan``.
    """
    if _classifier_instance is None:
        raise RuntimeError("Classifier not initialised — check lifespan.")
    return _classifier_instance


def provide_settings() -> Settings:
    """FastAPI dependency: return the singleton :class:`Settings` instance.

    Returns:
        The global :class:`Settings` object.
    """
    return get_settings()


def get_uptime() -> float:
    """Return seconds elapsed since the application started.

    Returns:
        Float number of seconds since ``lifespan`` recorded startup.
    """
    return time.monotonic() - _startup_time
