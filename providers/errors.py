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


# Statuses produced by raise_provider_error() for transient, provider-side
# conditions: rate limiting, unavailability, gateway errors, timeouts.
_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def is_transient_provider_error(exc: HTTPException) -> bool:
    """Whether a provider failure is worth retrying on another provider.

    Client/request errors (400, 404, ...) and credential/permission errors
    (401, 403) are not: another provider won't fix a malformed request, and
    a misconfigured key should surface rather than be masked by fallback.
    An unrecognized upstream 4xx is mapped to 502 for the client, but is
    still a request problem, so the original upstream status is checked too.
    """
    if exc.status_code not in _TRANSIENT_STATUS_CODES:
        return False

    cause = exc.__cause__
    if isinstance(cause, httpx.HTTPStatusError):
        upstream_status = cause.response.status_code
        return upstream_status == 429 or upstream_status >= 500

    return True


def is_timeout_provider_error(exc: HTTPException) -> bool:
    """Whether a provider failure was Azir giving up waiting on the upstream
    (as opposed to a fast error response or a failure to connect at all).
    """
    return isinstance(exc.__cause__, httpx.TimeoutException)
