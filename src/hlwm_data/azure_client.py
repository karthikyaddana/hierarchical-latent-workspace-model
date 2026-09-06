from __future__ import annotations

import json
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional

from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI, RateLimitError

from .util import append_jsonl


class SlidingWindowRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.limit = max(1, requests_per_minute)
        self.timestamps: Deque[float] = deque()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.timestamps and now - self.timestamps[0] >= 60.0:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.limit:
                    self.timestamps.append(now)
                    return
                wait_for = max(0.05, 60.0 - (now - self.timestamps[0]))
            time.sleep(min(wait_for, 2.0))


class BudgetExceededError(RuntimeError):
    """Raised before a provider attempt that would exceed the configured budget."""


@dataclass
class JsonCompletion:
    value: Dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    request_id: Optional[str]
    elapsed_seconds: float


@dataclass
class TextCompletion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    request_id: Optional[str]
    elapsed_seconds: float


def parse_json_object(text: str) -> Dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1] if "\n" in value else value[3:]
        if value.rstrip().endswith("```"):
            value = value.rstrip()[:-3]
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The model response did not contain a JSON object")
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("The model response JSON must be an object")
    return parsed


class AzureTeacherClient:
    _auth_lock = threading.Lock()
    _shared_credentials: Dict[str, Any] = {}
    _shared_token_providers: Dict[str, Any] = {}

    @classmethod
    def _shared_auth(cls, scope: str):
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        with cls._auth_lock:
            provider = cls._shared_token_providers.get(scope)
            if provider is not None:
                return provider
            credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
            provider = get_bearer_token_provider(credential, scope)
            cls._shared_credentials[scope] = credential
            cls._shared_token_providers[scope] = provider
            return provider

    def __init__(self, azure_config: Dict[str, Any], request_log: Path) -> None:
        token_provider = self._shared_auth(str(azure_config["scope"]))
        self._token_provider = token_provider
        self.client = OpenAI(
            base_url=azure_config["endpoint"],
            api_key=token_provider,
            timeout=float(azure_config.get("timeout_seconds", 180)),
            max_retries=0,
        )
        self.deployment = azure_config["deployment"]
        self.max_retries = int(azure_config.get("max_retries", 5))
        self.temperature = float(azure_config.get("temperature", 0.35))
        self.max_completion_tokens = int(azure_config.get("max_completion_tokens", 12000))
        self.rate_limiter = SlidingWindowRateLimiter(int(azure_config.get("requests_per_minute", 60)))
        self.request_log = request_log
        self._log_lock = threading.Lock()
        self._capability_lock = threading.Lock()
        self._structured = True
        self._token_parameter = "max_tokens"
        self._include_temperature = True

    def close(self) -> None:
        self.client.close()

    def warm_auth(self) -> None:
        """Acquire one cached token before a large worker pool starts."""
        self._token_provider()

    def _log(self, row: Dict[str, Any]) -> None:
        with self._log_lock:
            append_jsonl(self.request_log, row)

    def _create(
        self,
        system: str,
        user: str,
        structured: bool,
        token_parameter: str,
        include_temperature: bool,
    ):
        kwargs: Dict[str, Any] = {
            "model": self.deployment,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        kwargs[token_parameter] = self.max_completion_tokens
        if include_temperature:
            kwargs["temperature"] = self.temperature
        if structured:
            kwargs["response_format"] = {"type": "json_object"}
        return self.client.chat.completions.create(**kwargs)

    def chat_json(
        self,
        system: str,
        user: str,
        operation: str,
        item_id: str,
        *,
        before_attempt: Optional[Callable[[], None]] = None,
        record_usage: Optional[Callable[[int, int], None]] = None,
    ) -> JsonCompletion:
        with self._capability_lock:
            structured = self._structured
            token_parameter = self._token_parameter
            include_temperature = self._include_temperature
        last_error: Optional[BaseException] = None
        transport_attempt = 0
        compatibility_attempt = 0
        while transport_attempt <= self.max_retries and compatibility_attempt <= 4:
            self.rate_limiter.acquire()
            if before_attempt is not None:
                before_attempt()
            started = time.monotonic()
            try:
                response = self._create(
                    system,
                    user,
                    structured=structured,
                    token_parameter=token_parameter,
                    include_temperature=include_temperature,
                )
                elapsed = time.monotonic() - started
                message = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                if record_usage is not None:
                    record_usage(prompt_tokens, completion_tokens)
                value = parse_json_object(message)
                result = JsonCompletion(
                    value=value,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    request_id=getattr(response, "_request_id", None),
                    elapsed_seconds=elapsed,
                )
                self._log(
                    {
                        "operation": operation,
                        "item_id": item_id,
                        "status": "ok",
                        "attempt": transport_attempt + compatibility_attempt + 1,
                        "request_id": result.request_id,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "elapsed_seconds": round(elapsed, 3),
                    }
                )
                with self._capability_lock:
                    self._structured = structured
                    self._token_parameter = token_parameter
                    self._include_temperature = include_temperature
                return result
            except BadRequestError as exc:
                last_error = exc
                message = str(exc).lower()
                if "max_tokens" in message and "max_completion_tokens" in message:
                    token_parameter = "max_completion_tokens"
                    compatibility_attempt += 1
                    with self._capability_lock:
                        self._token_parameter = token_parameter
                    continue
                if "temperature" in message and (
                    "unsupported" in message or "only the default" in message
                ):
                    include_temperature = False
                    compatibility_attempt += 1
                    with self._capability_lock:
                        self._include_temperature = False
                    continue
                if structured:
                    structured = False
                    compatibility_attempt += 1
                    with self._capability_lock:
                        self._structured = False
                    continue
                break
            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                if status_code is not None and int(status_code) < 500:
                    break
            elapsed = time.monotonic() - started
            transport_attempt += 1
            self._log(
                {
                    "operation": operation,
                    "item_id": item_id,
                    "status": "retry",
                    "attempt": transport_attempt + compatibility_attempt,
                    "error_type": type(last_error).__name__ if last_error else "unknown",
                    "elapsed_seconds": round(elapsed, 3),
                }
            )
            if transport_attempt <= self.max_retries:
                time.sleep(min(60.0, (2 ** max(0, transport_attempt - 1)) + random.random()))
        self._log(
            {
                "operation": operation,
                "item_id": item_id,
                "status": "failed",
                "error_type": type(last_error).__name__ if last_error else "unknown",
                "error": str(last_error)[:500] if last_error else "unknown error",
            }
        )
        raise RuntimeError("Azure request failed for %s: %s" % (item_id, last_error))

    def chat_text(
        self,
        system: str,
        user: str,
        operation: str,
        item_id: str,
        *,
        max_completion_tokens: Optional[int] = None,
        before_attempt: Optional[Callable[[], None]] = None,
        record_usage: Optional[Callable[[int, int], None]] = None,
    ) -> TextCompletion:
        """Plain-text completion for artifact drafting; never requests JSON mode."""

        with self._capability_lock:
            token_parameter = self._token_parameter
            include_temperature = self._include_temperature
        token_budget = int(max_completion_tokens or self.max_completion_tokens)
        last_error: Optional[BaseException] = None
        transport_attempt = 0
        compatibility_attempt = 0
        while transport_attempt <= self.max_retries and compatibility_attempt <= 4:
            self.rate_limiter.acquire()
            if before_attempt is not None:
                before_attempt()
            started = time.monotonic()
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.deployment,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }
                kwargs[token_parameter] = token_budget
                if include_temperature:
                    kwargs["temperature"] = self.temperature
                response = self.client.chat.completions.create(**kwargs)
                elapsed = time.monotonic() - started
                message = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                if record_usage is not None:
                    record_usage(prompt_tokens, completion_tokens)
                result = TextCompletion(
                    text=message,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    request_id=getattr(response, "_request_id", None),
                    elapsed_seconds=elapsed,
                )
                self._log(
                    {
                        "operation": operation,
                        "item_id": item_id,
                        "status": "ok",
                        "attempt": transport_attempt + compatibility_attempt + 1,
                        "request_id": result.request_id,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "elapsed_seconds": round(elapsed, 3),
                    }
                )
                with self._capability_lock:
                    self._token_parameter = token_parameter
                    self._include_temperature = include_temperature
                return result
            except BadRequestError as exc:
                last_error = exc
                message = str(exc).lower()
                if "max_tokens" in message and "max_completion_tokens" in message:
                    token_parameter = "max_completion_tokens"
                    compatibility_attempt += 1
                    with self._capability_lock:
                        self._token_parameter = token_parameter
                    continue
                if "temperature" in message and (
                    "unsupported" in message or "only the default" in message
                ):
                    include_temperature = False
                    compatibility_attempt += 1
                    with self._capability_lock:
                        self._include_temperature = False
                    continue
                break
            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                if status_code is not None and int(status_code) < 500:
                    break
            elapsed = time.monotonic() - started
            transport_attempt += 1
            self._log(
                {
                    "operation": operation,
                    "item_id": item_id,
                    "status": "retry",
                    "attempt": transport_attempt + compatibility_attempt,
                    "error_type": type(last_error).__name__ if last_error else "unknown",
                    "elapsed_seconds": round(elapsed, 3),
                }
            )
            if transport_attempt <= self.max_retries:
                time.sleep(min(60.0, (2 ** max(0, transport_attempt - 1)) + random.random()))
        self._log(
            {
                "operation": operation,
                "item_id": item_id,
                "status": "failed",
                "error_type": type(last_error).__name__ if last_error else "unknown",
                "error": str(last_error)[:500] if last_error else "unknown error",
            }
        )
        raise RuntimeError("Azure text request failed for %s: %s" % (item_id, last_error))
