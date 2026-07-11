"""LLM client — single doorway for every chat call the product makes.

If Portkey is configured, calls route through the gateway (which records
latency, tokens, cost, and errors in the Portkey dashboard). If not,
calls go straight to Groq. Both clients speak the same OpenAI-style
interface, so callers never know the difference.
"""

from __future__ import annotations

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from backend.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    PORTKEY_API_KEY,
    PORTKEY_PROVIDER_SLUG,
)

_client = None
_model = None
_via = None  # "portkey" or "groq" — surfaced in /health


def _init():
    global _client, _model, _via
    if _client is not None:
        return

    if PORTKEY_API_KEY and PORTKEY_PROVIDER_SLUG:
        from portkey_ai import Portkey

        _client = Portkey(api_key=PORTKEY_API_KEY)
        # Portkey addresses models as "@<integration-slug>/<model-name>".
        _model = f"@{PORTKEY_PROVIDER_SLUG}/{GROQ_MODEL}"
        _via = "portkey"
    else:
        from groq import Groq

        _client = Groq(api_key=GROQ_API_KEY)
        _model = GROQ_MODEL
        _via = "groq"


def _is_rate_limit_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "429" in message or "rate_limit" in message or "rate limit" in message


# Every caller — planner, guardrail, responder, eval judge, RAGAS metrics —
# goes through this one function, so retry logic added here benefits all
# of them at once. Exponential backoff (not a fixed wait like the Gemini
# embedding retry) because Groq's TPM window is short-lived — a few
# seconds of backoff is usually enough to let tokens free up, unlike
# Gemini's 60-second embedding quota window.
@retry(
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(5),
    reraise=True,
)
def chat(messages: list[dict], temperature: float = 0.1, json_mode: bool = False) -> str:
    """Send a chat completion and return the reply text."""
    _init()
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = _client.chat.completions.create(
        model=_model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content


def gateway_status() -> str:
    _init()
    return _via
