"""Model-endpoint configuration shared by the OPIM agents.

All agents call an OpenAI-compatible ``chat/completions`` endpoint.  The
default is Google's official OpenAI-compatible Gemini endpoint; override it
with ``OPIM_API_ENDPOINT`` for any other compatible provider.  No third-party
relay is used by default, so data and keys only go where the user configured.
"""
import os

DEFAULT_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
# The paper uses Gemini-2.0-flash for both text and image scoring (Sec. 3.1, 4.3).
DEFAULT_MODEL = "gemini-2.0-flash"


def api_endpoint() -> str:
    return os.getenv("OPIM_API_ENDPOINT", DEFAULT_API_ENDPOINT).strip()


def api_model(default: str = DEFAULT_MODEL) -> str:
    return os.getenv("OPIM_API_MODEL", default).strip() or default


def api_key() -> str:
    return os.getenv("OPIM_API_KEY", "").strip()
