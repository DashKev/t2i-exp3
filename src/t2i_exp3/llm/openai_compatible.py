from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence
from urllib import error, request


@dataclass(frozen=True)
class ChatResponse:
    text: str
    model: str | None
    usage: dict[str, Any]
    raw: dict[str, Any]


class OpenAICompatibleClient:
    """Small dependency-free client for OpenAI-compatible local LLM services."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 300.0,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.extra_headers = dict(extra_headers or {})

    def list_models(self) -> list[str]:
        raw = self._request("GET", "/v1/models")
        data = raw.get("data", [])
        if not isinstance(data, list):
            raise RuntimeError("/v1/models 返回格式不符合 OpenAI-compatible 约定")
        models = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                models.append(item["id"])
        return models

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> ChatResponse:
        chosen_model = model or self.model
        if not chosen_model:
            models = self.list_models()
            if not models:
                raise RuntimeError("LLM 服务没有返回可用模型")
            chosen_model = models[0]
        body: dict[str, Any] = {
            "model": chosen_model,
            "messages": [dict(message) for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if seed is not None:
            body["seed"] = seed
        if extra_body:
            body.update(dict(extra_body))
        raw = self._request("POST", "/v1/chat/completions", body)
        choices = raw.get("choices", [])
        if not choices or not isinstance(choices, list):
            raise RuntimeError("/v1/chat/completions 返回缺少 choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError("/v1/chat/completions choices[0] 格式错误")
        message = first.get("message", {})
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("/v1/chat/completions 返回缺少 message.content")
        usage = raw.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        return ChatResponse(
            text=message["content"].strip(),
            model=raw.get("model") if isinstance(raw.get("model"), str) else chosen_model,
            usage=usage,
            raw=raw,
        )

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def _request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Accept": "application/json",
            **self.extra_headers,
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(f"{self.base_url}{path}", data=payload, method=method, headers=headers)
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"LLM HTTP {exc.code} {path}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"LLM 请求失败 {path}: {exc.reason}") from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM 返回不是 JSON：{raw[:200]}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("LLM 返回 JSON 不是对象")
        return parsed
