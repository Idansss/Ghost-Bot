class BotError(Exception):
    """Base bot error."""


class UpstreamError(BotError):
    """Raised when an upstream API is unavailable."""


class ValidationError(BotError):
    """Raised for invalid user input."""
