"""PaperAlchemy request logging middleware and global exception handlers."""

from __future__ import annotations

import logging
import time
import traceback
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.exceptions import PaperAlchemyError
from src.schemas.api.error import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)

# Paths excluded from request logging
EXCLUDED_PATHS = {"/api/v1/ping", "/docs", "/openapi.json", "/redoc"}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs requests with timing and adds X-Request-ID header.

    Also catches unhandled exceptions that bypass FastAPI's exception handlers
    (a known Starlette BaseHTTPMiddleware limitation).
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate or forward request ID
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            # Catch unhandled exceptions that bypass Starlette's exception handlers
            duration_ms = (time.perf_counter() - start) * 1000
            response = _build_unhandled_error_response(request, exc, request_id)
            if request.url.path not in EXCLUDED_PATHS:
                logger.info(
                    "%s %s -> %s (%.1fms) [request_id=%s]",
                    request.method,
                    request.url.path,
                    response.status_code,
                    duration_ms,
                    request_id,
                )
            return response

        duration_ms = (time.perf_counter() - start) * 1000

        # Add request ID to response
        response.headers["X-Request-ID"] = request_id

        # Log unless excluded
        if request.url.path not in EXCLUDED_PATHS:
            logger.info(
                "%s %s -> %s (%.1fms) [request_id=%s]",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
                request_id,
            )

        return response


def _build_unhandled_error_response(request: Request, exc: Exception, request_id: str) -> JSONResponse:
    """Build a 500 JSON response for unhandled exceptions."""
    logger.error(
        "%s %s raised unhandled %s: %s [request_id=%s]",
        request.method,
        request.url.path,
        type(exc).__name__,
        str(exc),
        request_id,
        exc_info=True,
    )
    detail_dict = None
    if getattr(request.app, "debug", False):
        detail_dict = {"traceback": traceback.format_exc()}

    body = ErrorResponse(
        error=ErrorDetail(
            type="InternalServerError",
            message="Internal Server Error",
            request_id=request_id,
            detail=detail_dict,
        )
    )
    response = JSONResponse(status_code=500, content=body.model_dump())
    response.headers["X-Request-ID"] = request_id
    return response


def _get_request_id(request: Request) -> str | None:
    """Extract request_id from request state if available."""
    return getattr(request.state, "request_id", None)


async def paper_alchemy_error_handler(request: Request, exc: PaperAlchemyError) -> JSONResponse:
    """Handle all PaperAlchemyError subclasses."""
    request_id = _get_request_id(request)
    logger.error(
        "%s %s raised %s: %s [request_id=%s]",
        request.method,
        request.url.path,
        type(exc).__name__,
        exc.detail,
        request_id,
    )
    body = ErrorResponse(
        error=ErrorDetail(
            type=type(exc).__name__,
            message=exc.detail,
            request_id=request_id,
        )
    )
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions — never leak internals in production."""
    request_id = _get_request_id(request)
    logger.error(
        "%s %s raised unhandled %s: %s [request_id=%s]",
        request.method,
        request.url.path,
        type(exc).__name__,
        str(exc),
        request_id,
        exc_info=True,
    )
    detail_dict = None
    if getattr(request.app, "debug", False):
        detail_dict = {"traceback": traceback.format_exc()}

    body = ErrorResponse(
        error=ErrorDetail(
            type="InternalServerError",
            message="Internal Server Error",
            request_id=request_id,
            detail=detail_dict,
        )
    )
    return JSONResponse(status_code=500, content=body.model_dump())


def register_error_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the app."""
    app.add_exception_handler(PaperAlchemyError, paper_alchemy_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
