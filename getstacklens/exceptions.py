"""StackLens SDK exceptions."""


class StackLensError(Exception):
    """Base exception for all StackLens SDK errors."""


class ConfigurationError(StackLensError):
    """Raised when the SDK is not configured. Call stacklens.configure() first."""


class AuthError(StackLensError):
    """Raised when the API key is invalid or does not have the required scope."""


class ApiError(StackLensError):
    """Raised when the StackLens API returns an error response."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"API error {status_code}: {message}")


class NetworkError(StackLensError):
    """Raised when the StackLens API cannot be reached (timeout, DNS, connection refused)."""
