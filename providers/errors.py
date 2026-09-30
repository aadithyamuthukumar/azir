import httpx
from fastapi import HTTPException

_UPSTREAM_STATUS_DETAIL = {
    400: "sent an invalid request",
    401: "rejected the configured API credentials",
    403: "denied permission for this request",
    404: "could not find the requested model or resource",
    429: "rate-limited this request",
}


def raise_provider_error(exc: Exception, provider: str) -> None:
    """Translate an httpx failure into a clean FastAPI HTTPException.

    Only a generic, templated message is used so raw provider response
    bodies, headers, and API keys never reach the client.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        reason = _UPSTREAM_STATUS_DETAIL.get(status_code)

        if reason:
            raise HTTPException(
                status_code=status_code,
                detail=f"{provider} {reason}.",
            ) from exc

        if status_code >= 500:
            raise HTTPException(
                status_code=502,
                detail=f"{provider} is currently unavailable. Please try again later.",
            ) from exc

        raise HTTPException(
            status_code=502,
            detail=f"{provider} returned an unexpected error.",
        ) from exc

    if isinstance(exc, httpx.TimeoutException):
        raise HTTPException(
            status_code=504,
            detail=f"Timed out waiting for a response from {provider}.",
        ) from exc

    if isinstance(exc, httpx.RequestError):
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach {provider}.",
        ) from exc

    raise exc
