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

# Errors worth retrying: rate limiting and transient server or network trouble.
# Everything else - a bad key, an unknown model, a malformed request - is a real
# failure that retrying only delays.
RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

MAX_ATTEMPTS = 5
BASE_DELAY_SECONDS = 2.0


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
