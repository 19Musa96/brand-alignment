"""Exponential-backoff retry logic for Gemini API calls.

Wraps API calls with tenacity-based retries that handle:
- 429 rate-limit responses
- Transient 5xx server errors

Configuration via environment variables:
- GEMINI_MAX_RETRIES: Maximum number of retry attempts (default: 5)
- GEMINI_BASE_DELAY: Base delay in seconds for exponential backoff (default: 1.0)
"""

import logging
import os

from google.genai.errors import ClientError, ServerError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

_RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retriable(exc: BaseException) -> bool:
    """Return True if the exception is a retriable Gemini API error."""
    if isinstance(exc, ServerError):
        return True
    if isinstance(exc, ClientError) and getattr(exc, "code", None) == 429:
        return True
    return False


def _get_max_retries() -> int:
    raw = os.getenv("GEMINI_MAX_RETRIES", "5")
    try:
        return max(1, int(raw))
    except (ValueError, TypeError):
        return 5


def _get_base_delay() -> float:
    raw = os.getenv("GEMINI_BASE_DELAY", "1.0")
    try:
        return max(0.1, float(raw))
    except (ValueError, TypeError):
        return 1.0


def gemini_retry(func):
    """Decorator that adds exponential-backoff retry logic to a function.

    Retries on 429 rate-limit and 5xx server errors from the Gemini API.
    Uses configurable max_retries and base_delay from environment variables.
    """
    max_retries = _get_max_retries()
    base_delay = _get_base_delay()

    decorated = retry(
        retry=retry_if_exception(_is_retriable),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=base_delay, min=base_delay, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)

    return decorated
