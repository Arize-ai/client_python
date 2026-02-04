"""Evaluator-specific exception classes."""


class ArizeException(Exception):
    """Base exception for Arize experiment evaluator errors."""


class ArizeContextLimitExceeded(ArizeException):
    """Raised when context limit is exceeded during evaluation."""


class ArizeTemplateMappingError(ArizeException):
    """Raised when template mapping fails during evaluation."""
