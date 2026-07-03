"""LLM client — single doorway for every chat call the product makes.

If Portkey is configured, calls route through the gateway (which records
latency, tokens, cost, and errors in the Portkey dashboard). If not,
calls go straight to Groq. Both clients speak the same OpenAI-style
interface, so callers never know the difference.
"""

from __future__ import annotations

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
