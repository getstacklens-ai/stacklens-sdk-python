"""GetStackLens SDK exceptions."""


class GetStackLensError(Exception):
    """Base exception for all GetStackLens SDK errors."""


class ConfigurationError(GetStackLensError):
    """Raised when the SDK is not configured. Call getstacklens.configure() first."""


class AuthError(GetStackLensError):
    """Raised when the API key is invalid or does not have the required scope."""


class ApiError(GetStackLensError):
    """Raised when the GetStackLens API returns an error response."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"API error {status_code}: {message}")


class NetworkError(GetStackLensError):
    """Raised when the GetStackLens API cannot be reached (timeout, DNS, connection refused)."""
