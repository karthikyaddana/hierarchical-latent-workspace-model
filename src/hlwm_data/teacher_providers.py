from __future__ import annotations

import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI, RateLimitError

from .azure_client import (
    AzureTeacherClient,
    JsonCompletion,
    SlidingWindowRateLimiter,
    TextCompletion,
    parse_json_object,
)
from .util import append_jsonl


@dataclass(frozen=True)
class ProviderStatus:
    provider_id: str
    model: str
    provider_type: str
    available: bool
    reason: str
    roles: tuple[str, ...]


class TeacherProvider:
    provider_id: str
    model: str
    roles: tuple[str, ...]

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
        raise NotImplementedError

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
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class AzureProvider(TeacherProvider):
    def __init__(
        self,
        provider_id: str,
        model: str,
        roles: Iterable[str],
        azure_config: Mapping[str, Any],
        request_log: Path,
    ) -> None:
        config = dict(azure_config)
        config["deployment"] = model
        self.provider_id = provider_id
        self.model = model
        self.roles = tuple(str(role) for role in roles)
        self._client = AzureTeacherClient(config, request_log)

    def warm_auth(self) -> None:
        self._client.warm_auth()

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
        return self._client.chat_json(
            system,
            user,
            operation,
            item_id,
            before_attempt=before_attempt,
            record_usage=record_usage,
        )

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
        return self._client.chat_text(
            system,
            user,
            operation,
            item_id,
            max_completion_tokens=max_completion_tokens,
            before_attempt=before_attempt,
            record_usage=record_usage,
        )

    def close(self) -> None:
        self._client.close()


class OpenAICompatibleProvider(TeacherProvider):
    """Small JSON client for NIM, Ollama and other OpenAI-compatible endpoints."""

    def __init__(
        self,
        provider_id: str,
        model: str,
        roles: Iterable[str],
        config: Mapping[str, Any],
        request_log: Path,
    ) -> None:
        self.provider_id = provider_id
        self.model = model
        self.roles = tuple(str(role) for role in roles)
        self.base_url = str(config["base_url"]).rstrip("/")
        api_key_env = str(config.get("api_key_env") or "")
        api_key = os.getenv(api_key_env) if api_key_env else str(config.get("api_key") or "local")
        if not api_key:
            raise ValueError("missing API key environment variable %s" % api_key_env)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=float(config.get("timeout_seconds", 180)),
            max_retries=0,
        )
        self.max_retries = int(config.get("max_retries", 3))
        self.max_completion_tokens = int(config.get("max_completion_tokens", 6000))
        self.temperature = float(config.get("temperature", 0.2))
        self.top_p = config.get("top_p")
        self.extra_body = dict(config.get("extra_body") or {})
        self.stream_text = bool(config.get("stream", False))
        self.rate_limiter = SlidingWindowRateLimiter(
            int(config.get("requests_per_minute", 30))
        )
        self.request_log = request_log
        self._log_lock = threading.Lock()
        self._structured = True
        self._token_parameter = "max_tokens"
        self._include_stream_options = True

    def _log(self, row: Dict[str, Any]) -> None:
        row = dict(row)
        row.update({"provider_id": self.provider_id, "model": self.model})
        with self._log_lock:
            append_jsonl(self.request_log, row)

    def _create(
        self,
        system: str,
        user: str,
        structured: bool,
        token_parameter: str,
        *,
        stream: bool = False,
        max_completion_tokens: Optional[int] = None,
    ):
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            kwargs["top_p"] = float(self.top_p)
        kwargs[token_parameter] = int(max_completion_tokens or self.max_completion_tokens)
        if structured:
            kwargs["response_format"] = {"type": "json_object"}
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if stream:
            kwargs["stream"] = True
            if self._include_stream_options:
                kwargs["stream_options"] = {"include_usage": True}
        return self.client.chat.completions.create(**kwargs)

    def _collect_stream(self, response) -> tuple[str, int, int]:
        """Assemble streamed content; reasoning deltas are read past and dropped.

        Provider thinking traces (``reasoning_content``) are never persisted or
        returned: private chain-of-thought is not a permitted training target.
        """

        pieces: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        for chunk in response:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                pieces.append(content)
        return "".join(pieces), prompt_tokens, completion_tokens

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
        structured = self._structured
        token_parameter = self._token_parameter
        last_error: Optional[BaseException] = None
        compatibility_attempts = 0
        for attempt in range(self.max_retries + 1):
            self.rate_limiter.acquire()
            if before_attempt is not None:
                before_attempt()
            started = time.monotonic()
            try:
                response = self._create(system, user, structured, token_parameter)
                elapsed = time.monotonic() - started
                content = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                if record_usage is not None:
                    record_usage(prompt_tokens, completion_tokens)
                completion = JsonCompletion(
                    value=parse_json_object(content),
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
                        "attempt": attempt + compatibility_attempts + 1,
                        "request_id": completion.request_id,
                        "prompt_tokens": completion.prompt_tokens,
                        "completion_tokens": completion.completion_tokens,
                        "elapsed_seconds": round(elapsed, 3),
                    }
                )
                self._structured = structured
                self._token_parameter = token_parameter
                return completion
            except BadRequestError as exc:
                last_error = exc
                message = str(exc).lower()
                if "max_tokens" in message and "max_completion_tokens" in message:
                    token_parameter = "max_completion_tokens"
                    compatibility_attempts += 1
                    continue
                if structured:
                    structured = False
                    compatibility_attempts += 1
                    continue
                break
            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                last_error = exc
            except Exception as exc:  # Provider-specific transport errors.
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                if status_code is not None and int(status_code) < 500:
                    break
            self._log(
                {
                    "operation": operation,
                    "item_id": item_id,
                    "status": "retry",
                    "attempt": attempt + compatibility_attempts + 1,
                    "error_type": type(last_error).__name__ if last_error else "unknown",
                }
            )
            if attempt < self.max_retries:
                time.sleep(min(30.0, 2**attempt + random.random()))
        self._log(
            {
                "operation": operation,
                "item_id": item_id,
                "status": "failed",
                "error_type": type(last_error).__name__ if last_error else "unknown",
                "error": str(last_error)[:500] if last_error else "unknown error",
            }
        )
        raise RuntimeError("Provider request failed for %s: %s" % (item_id, last_error))

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
        """Plain-text completion; streams when configured so long artifact
        drafts survive gateway idle timeouts. Reasoning deltas are discarded."""

        token_parameter = self._token_parameter
        stream = self.stream_text
        last_error: Optional[BaseException] = None
        compatibility_attempts = 0
        for attempt in range(self.max_retries + 1):
            self.rate_limiter.acquire()
            if before_attempt is not None:
                before_attempt()
            started = time.monotonic()
            try:
                response = self._create(
                    system,
                    user,
                    structured=False,
                    token_parameter=token_parameter,
                    stream=stream,
                    max_completion_tokens=max_completion_tokens,
                )
                if stream:
                    content, prompt_tokens, completion_tokens = self._collect_stream(response)
                    request_id = None
                else:
                    content = response.choices[0].message.content or ""
                    usage = getattr(response, "usage", None)
                    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                    request_id = getattr(response, "_request_id", None)
                elapsed = time.monotonic() - started
                if record_usage is not None:
                    record_usage(prompt_tokens, completion_tokens)
                completion = TextCompletion(
                    text=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    request_id=request_id,
                    elapsed_seconds=elapsed,
                )
                self._log(
                    {
                        "operation": operation,
                        "item_id": item_id,
                        "status": "ok",
                        "attempt": attempt + compatibility_attempts + 1,
                        "request_id": completion.request_id,
                        "prompt_tokens": completion.prompt_tokens,
                        "completion_tokens": completion.completion_tokens,
                        "elapsed_seconds": round(elapsed, 3),
                    }
                )
                self._token_parameter = token_parameter
                return completion
            except BadRequestError as exc:
                last_error = exc
                message = str(exc).lower()
                if "max_tokens" in message and "max_completion_tokens" in message:
                    token_parameter = "max_completion_tokens"
                    compatibility_attempts += 1
                    continue
                if "stream_options" in message and self._include_stream_options:
                    self._include_stream_options = False
                    compatibility_attempts += 1
                    continue
                if stream and "stream" in message:
                    stream = False
                    compatibility_attempts += 1
                    continue
                break
            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                last_error = exc
            except Exception as exc:  # Provider-specific transport errors.
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                if status_code is not None and int(status_code) < 500:
                    break
            self._log(
                {
                    "operation": operation,
                    "item_id": item_id,
                    "status": "retry",
                    "attempt": attempt + compatibility_attempts + 1,
                    "error_type": type(last_error).__name__ if last_error else "unknown",
                }
            )
            if attempt < self.max_retries:
                time.sleep(min(30.0, 2**attempt + random.random()))
        self._log(
            {
                "operation": operation,
                "item_id": item_id,
                "status": "failed",
                "error_type": type(last_error).__name__ if last_error else "unknown",
                "error": str(last_error)[:500] if last_error else "unknown error",
            }
        )
        raise RuntimeError("Provider text request failed for %s: %s" % (item_id, last_error))

    def close(self) -> None:
        self.client.close()


def _endpoint_reachable(base_url: str, timeout: float = 0.5) -> bool:
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def provider_statuses(config: Mapping[str, Any]) -> List[ProviderStatus]:
    factory = config.get("teacher_factory") or {}
    statuses: List[ProviderStatus] = []
    for provider_id, spec_value in (factory.get("providers") or {}).items():
        spec = dict(spec_value or {})
        provider_type = str(spec.get("type") or "azure")
        model = str(spec.get("model") or spec.get("deployment") or "")
        roles = tuple(str(role) for role in spec.get("roles") or ())
        enabled = bool(spec.get("enabled", True))
        available = enabled
        reason = "configured"
        if not enabled:
            available, reason = False, "disabled"
        elif provider_type == "openai_compatible":
            env_name = str(spec.get("api_key_env") or "")
            if env_name and not os.getenv(env_name):
                available, reason = False, "missing %s" % env_name
            elif bool(spec.get("probe_local", False)) and not _endpoint_reachable(
                str(spec.get("base_url") or "")
            ):
                available, reason = False, "endpoint unavailable"
        elif provider_type != "azure":
            available, reason = False, "unsupported provider type"
        statuses.append(
            ProviderStatus(
                provider_id=str(provider_id),
                model=model,
                provider_type=provider_type,
                available=available,
                reason=reason,
                roles=roles,
            )
        )
    return statuses


def build_teacher_providers(
    config: Mapping[str, Any], request_log: Path
) -> Dict[str, TeacherProvider]:
    statuses = {status.provider_id: status for status in provider_statuses(config)}
    factory = config.get("teacher_factory") or {}
    result: Dict[str, TeacherProvider] = {}
    for provider_id, spec_value in (factory.get("providers") or {}).items():
        status = statuses[str(provider_id)]
        if not status.available:
            continue
        spec = dict(spec_value or {})
        if status.provider_type == "azure":
            provider = AzureProvider(
                status.provider_id,
                status.model,
                status.roles,
                config["azure"],
                request_log,
            )
        else:
            provider = OpenAICompatibleProvider(
                status.provider_id,
                status.model,
                status.roles,
                spec,
                request_log,
            )
        result[status.provider_id] = provider
    return result
