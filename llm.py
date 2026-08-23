"""
The shared path for every Groq call.

Both generation and grading go through here, so retries and timing are defined
once. The Phase 2 sweep makes roughly 120 calls in a row and will meet free-tier
rate limits; without a retry the sweep dies partway and leaves the corpus
ingested at whatever chunk size it had reached.
"""

import random
import time

from groq import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError


class DailyQuotaExceeded(RuntimeError):
    """
    The provider's daily token or request cap is exhausted.

    Distinct from a transient rate limit: no amount of waiting inside this
    process will clear it, so callers that can checkpoint their progress should
    stop cleanly rather than retry.
    """

# Errors worth retrying: rate limiting and transient server or network trouble.
# Everything else - a bad key, an unknown model, a malformed request - is a real
# failure that retrying only delays.
RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

MAX_ATTEMPTS = 5
BASE_DELAY_SECONDS = 2.0

# Groq enforces two kinds of rate limit and they need opposite responses. A
# per-minute limit clears in seconds and is worth backing off against. A daily
# token cap does not clear for hours, so retrying it five times over thirty
# seconds just turns one clear failure into five slow ones - which is exactly
# what happened to the first full sweep.
DAILY_LIMIT_MARKERS = ("tokens per day", "TPD", "requests per day", "RPD")


def is_daily_limit(error):
    """True when a rate limit is a daily quota rather than a per-minute one."""
    message = str(error)
    return any(marker in message for marker in DAILY_LIMIT_MARKERS)


def chat_completion(client, model, prompt, temperature):
    """
    Send one prompt and return `(response, latency_seconds)`.

    Latency is wall-clock time for the attempt that succeeded, so backoff sleeps
    from earlier attempts are excluded - a retried call should not be recorded
    as a slow one.
    """
    last_error = None

    for attempt in range(MAX_ATTEMPTS):
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response, time.perf_counter() - started
        except RETRYABLE as exc:
            last_error = exc

            # A daily quota will not clear within any sensible backoff. Surface
            # it immediately with the provider's own reset hint intact, so the
            # caller sees what to do rather than a generic "failed after 5
            # attempts" thirty seconds later.
            if isinstance(exc, RateLimitError) and is_daily_limit(exc):
                raise DailyQuotaExceeded(str(exc)) from exc

            if attempt == MAX_ATTEMPTS - 1:
                break
            # Exponential backoff with jitter, so a sweep's parallel-ish retries
            # do not all come back at the same instant.
            delay = BASE_DELAY_SECONDS * (2**attempt) + random.uniform(0, 1)
            print(f"    (Groq {type(exc).__name__}, retrying in {delay:.1f}s)")
            time.sleep(delay)

    raise RuntimeError(
        f"Groq call failed after {MAX_ATTEMPTS} attempts: {type(last_error).__name__}: {last_error}"
    ) from last_error
