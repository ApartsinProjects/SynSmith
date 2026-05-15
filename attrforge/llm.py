"""Backend-agnostic LLM client.

We deliberately keep the surface tiny: a single ``chat()`` call that takes a
system prompt and a list of (role, content) messages and returns plain text.
A small ``json_chat()`` helper does best-effort JSON parsing with a single
retry. This keeps the rest of the codebase decoupled from any specific
provider SDK and lets us swap OpenAI for Anthropic or a local model by
flipping one config key.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

Role = Literal["system", "user", "assistant"]
Message = dict[str, str]


class LLMError(RuntimeError):
    """Raised when an LLM call fails after all retries."""


class LLMClient(Protocol):
    """Minimal interface every backend must implement."""

    def chat(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str: ...


@dataclass
class LLMConfig:
    """Backend selector plus model and sampling defaults."""

    backend: Literal["openai", "anthropic", "echo"] = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
    extra: dict[str, Any] = field(default_factory=dict)


class OpenAIClient:
    """Thin wrapper around the official ``openai`` SDK.

    Imported lazily so the dependency stays optional.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "openai package is required for the openai backend. "
                "Install with `pip install attrforge[openai]`."
            ) from exc
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise LLMError(f"Environment variable {cfg.api_key_env} is not set.")
        self.client = OpenAI(api_key=api_key, base_url=cfg.base_url)
        self.cfg = cfg

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def chat(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        full = [{"role": "system", "content": system}, *messages]
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=full,
            temperature=temperature if temperature is not None else self.cfg.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.cfg.max_tokens,
        )
        return resp.choices[0].message.content or ""


class AnthropicClient:
    """Thin wrapper around the official ``anthropic`` SDK."""

    def __init__(self, cfg: LLMConfig) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise LLMError(
                "anthropic package is required for the anthropic backend. "
                "Install with `pip install attrforge[anthropic]`."
            ) from exc
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise LLMError(f"Environment variable {cfg.api_key_env} is not set.")
        self.client = Anthropic(api_key=api_key, base_url=cfg.base_url)
        self.cfg = cfg

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def chat(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        resp = self.client.messages.create(
            model=self.cfg.model,
            system=system,
            messages=messages,
            temperature=temperature if temperature is not None else self.cfg.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.cfg.max_tokens,
        )
        parts = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)


class EchoClient:
    """Offline backend that returns a canned response. Used for tests and demos.

    The echo client doesn't call any network. It tries to satisfy structured
    output requests by returning a syntactically valid JSON object derived
    from any ``Target attributes`` or ``sample_id`` hints found in the prompt.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._counter = 0

    def chat(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        last = messages[-1]["content"] if messages else ""
        self._counter += 1
        sid = f"echo_{self._counter:03d}"
        if "Output JSON" in system or "Output JSON" in last or '"sample_id"' in last:
            attrs = _extract_attrs(last)
            return json.dumps(
                {
                    "sample_id": sid,
                    "text": f"[echo sample {self._counter}] placeholder text.",
                    "attributes": attrs,
                }
            )
        if "rewrite" in system.lower() or "improving" in system.lower():
            return "Generate diverse, realistic synthetic examples that match the requested attributes."
        return f"[echo response {self._counter}]"


def _extract_attrs(text: str) -> dict[str, str]:
    """Pull a JSON object of attributes out of a prompt body if present."""
    m = re.search(r"\{[^{}]*\}", text)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return {k: str(v) for k, v in obj.items() if isinstance(v, (str, int, float))}
    except Exception:
        return {}


def build_client(cfg: LLMConfig) -> LLMClient:
    """Construct a client for the configured backend."""
    backend = cfg.backend.lower()
    if backend == "openai":
        return OpenAIClient(cfg)
    if backend == "anthropic":
        return AnthropicClient(cfg)
    if backend == "echo":
        return EchoClient(cfg)
    raise LLMError(f"Unknown backend: {cfg.backend!r}")


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_JSON_BLOB = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def parse_json(text: str) -> Any:
    """Best-effort JSON parser that tolerates code fences and leading prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE.search(text)
    if m:
        return json.loads(m.group(1))
    m = _JSON_BLOB.search(text)
    if m:
        return json.loads(m.group(1))
    raise ValueError(f"could not parse JSON from response: {text[:200]!r}")


def json_chat(
    client: LLMClient,
    system: str,
    messages: list[Message],
    *,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    retries: int = 1,
) -> Any:
    """Call ``client.chat`` and parse the response as JSON, with one repair retry.

    The repair retry appends a brief instruction reminding the model to return
    valid JSON, which empirically fixes most malformed outputs without
    inflating cost.
    """
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        raw = client.chat(
            system, messages, temperature=temperature, max_tokens=max_tokens
        )
        try:
            return parse_json(raw)
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "Your previous response could not be parsed as JSON. "
                            "Return ONLY a valid JSON object, with no prose or code fences."
                        ),
                    },
                ]
                time.sleep(0.5)
    raise LLMError(f"json_chat failed to produce valid JSON: {last_err}")
