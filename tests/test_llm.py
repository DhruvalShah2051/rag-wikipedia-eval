"""
Tests for the shared Groq call path.

The sweep makes roughly 120 calls in sequence and will meet free-tier rate
limits. If a retry is missing or retries the wrong thing, the sweep dies partway
through and leaves the corpus ingested at whichever chunk size it had reached -
so the retry policy is worth pinning precisely.
"""

import httpx
import pytest
from groq import APIConnectionError, AuthenticationError, RateLimitError

import llm
from conftest import FakeGroqClient


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """Backoff is real seconds; tests should not actually wait them out."""
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)


def rate_limit_error():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError("rate limited", response=response, body=None)


def auth_error():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(401, request=request)
    return AuthenticationError("bad key", response=response, body=None)


def test_successful_call_returns_response_and_latency():
    client = FakeGroqClient("hello")

    response, latency = llm.chat_completion(client, "some-model", "a prompt", 0.1)

    assert response.choices[0].message.content == "hello"
    assert latency >= 0.0


def test_call_passes_model_prompt_and_temperature_through():
    client = FakeGroqClient("hello")

    llm.chat_completion(client, "some-model", "a prompt", 0.42)

    call = client.calls[0]
    assert call["model"] == "some-model"
    assert call["temperature"] == 0.42
    assert call["messages"] == [{"role": "user", "content": "a prompt"}]


def test_rate_limit_is_retried_and_then_succeeds():
    client = FakeGroqClient("hello", errors=[rate_limit_error(), rate_limit_error()])

    response, _latency = llm.chat_completion(client, "m", "p", 0.0)

    assert response.choices[0].message.content == "hello"
    assert len(client.calls) == 3  # two failures, then success


def test_connection_error_is_retried():
    request = httpx.Request("POST", "https://api.groq.com/")
    client = FakeGroqClient("hello", errors=[APIConnectionError(request=request)])

    llm.chat_completion(client, "m", "p", 0.0)

    assert len(client.calls) == 2


def test_authentication_error_is_not_retried():
    """
    A bad key will never succeed. Retrying it five times with backoff turns an
    instant, clear failure into a slow, confusing one.
    """
    client = FakeGroqClient("hello", errors=[auth_error()])

    with pytest.raises(AuthenticationError):
        llm.chat_completion(client, "m", "p", 0.0)

    assert len(client.calls) == 1


def daily_limit_error():
    """Groq's actual daily-cap message, which the first full sweep died on."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError(
        "Rate limit reached for model `openai/gpt-oss-20b` on tokens per day (TPD): "
        "Limit 200000, Used 198428, Requested 5651. Please try again in 29m22.128s.",
        response=response,
        body=None,
    )


def test_daily_quota_fails_fast_instead_of_retrying():
    """
    A per-minute limit clears in seconds; a daily cap does not clear for hours.
    Backing off five times over thirty seconds against a daily cap turns one
    clear failure into five slow ones and still fails.
    """
    client = FakeGroqClient("hello", errors=[daily_limit_error()])

    with pytest.raises(llm.DailyQuotaExceeded):
        llm.chat_completion(client, "m", "p", 0.0)

    assert len(client.calls) == 1, "daily quota must not be retried"


def test_daily_quota_error_keeps_the_providers_reset_hint():
    """The caller needs to know when it is worth trying again."""
    client = FakeGroqClient("hello", errors=[daily_limit_error()])

    with pytest.raises(llm.DailyQuotaExceeded, match="29m22"):
        llm.chat_completion(client, "m", "p", 0.0)


def test_per_minute_limit_is_still_retried():
    """The distinction must not swallow ordinary rate limits."""
    client = FakeGroqClient("hello", errors=[rate_limit_error()])

    llm.chat_completion(client, "m", "p", 0.0)

    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "message, expected",
    [
        ("limit reached on tokens per day (TPD): Limit 200000", True),
        ("Rate limit reached on requests per day (RPD)", True),
        ("Rate limit reached on tokens per minute (TPM)", False),
        ("Rate limit reached on requests per minute (RPM)", False),
    ],
)
def test_daily_limits_are_told_apart_from_per_minute_ones(message, expected):
    assert llm.is_daily_limit(message) is expected


def test_gives_up_after_max_attempts():
    client = FakeGroqClient("hello", errors=[rate_limit_error() for _ in range(llm.MAX_ATTEMPTS)])

    with pytest.raises(RuntimeError, match="failed after"):
        llm.chat_completion(client, "m", "p", 0.0)

    assert len(client.calls) == llm.MAX_ATTEMPTS


def test_latency_excludes_backoff_sleeps(monkeypatch):
    """
    A call that succeeded on its third attempt should be recorded at the
    duration of that attempt, not of the whole retry sequence. Otherwise a
    single rate limit would poison the mean latency metric.
    """
    slept = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))

    client = FakeGroqClient("hello", errors=[rate_limit_error(), rate_limit_error()])
    _response, latency = llm.chat_completion(client, "m", "p", 0.0)

    assert slept, "expected backoff to have been applied"
    assert latency < min(slept), "latency must not include the backoff waits"
