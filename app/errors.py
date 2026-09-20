"""Uniform, non-leaking error contract for the API (spec section 8)."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class ApiError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    message: str = "Something went wrong. Please try again."
    headers: dict[str, str] | None = None

    def __init__(
        self,
        message: str | None = None,
        *,
        status_code: int | None = None,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        if message is not None:
            self.message = message
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code
        if headers is not None:
            self.headers = headers
        super().__init__(self.message)


class AIUnavailableError(ApiError):
    status_code = 502
    code = "ai_unavailable"
    message = "Chat is temporarily unavailable. Please try again in a moment."


class AINotConfiguredError(ApiError):
    status_code = 503
    code = "service_not_configured"
    message = "Chat is not configured yet. Please contact the administrator."


class UnauthorizedError(ApiError):
    status_code = 401
    code = "unauthorized"
    message = "A valid session token is required for this chat session."


class NotFoundError(ApiError):
    status_code = 404
    code = "not_found"

    def __init__(self, message: str = "The requested resource was not found.") -> None:
        super().__init__(message=message)


class SessionNotFoundError(NotFoundError):
    code = "session_not_found"
    message = "That chat session could not be found. It may have expired."


class ForbiddenError(ApiError):
    status_code = 403
    code = "forbidden"
    message = "You do not have access to this conversation."


class RateLimitedError(ApiError):
    status_code = 429
    code = "rate_limited"
    message = "Too many requests. Please wait a moment and try again."
    headers = {"Retry-After": "60"}


_HTTP_FALLBACK_MESSAGES: dict[int, str] = {
    404: "The requested resource was not found.",
    405: "That HTTP method is not allowed here.",
    500: "Something went wrong. Please try again.",
}


def _error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def register_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.code, exc.message),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        logger.info("Validation error: %s", exc.errors())
        return JSONResponse(
            status_code=422,
            content=_error_body(
                "validation_error",
                "The request is invalid. Please provide a message between 1 and "
                "the maximum allowed characters.",
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = _HTTP_FALLBACK_MESSAGES.get(exc.status_code, "Request failed.")
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body("http_error", message),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server error")
        return JSONResponse(
            status_code=500,
            content=_error_body(
                "internal_error",
                "Something went wrong on our side. Please try again in a moment.",
            ),
        )
