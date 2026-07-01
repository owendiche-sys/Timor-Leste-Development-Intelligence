"""Minimal Ollama REST client with deterministic caching and JSON repair."""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json


@dataclass(frozen=True)
class GenerationResult:
    model: str
    response: str
    elapsed_seconds: float
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    cached: bool = False


class OllamaUnavailableError(RuntimeError):
    pass


def _remove_text_limits(value: Any) -> Any:
    """Avoid models filling schema maxLength fields with clipped prose."""

    if isinstance(value, dict):
        return {
            key: _remove_text_limits(item)
            for key, item in value.items()
            if key != "maxLength"
        }
    if isinstance(value, list):
        return [_remove_text_limits(item) for item in value]
    return value


def parse_json_response(text: str) -> Any:
    """Parse strict JSON, tolerating only common Markdown wrappers."""

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start_candidates = [position for position in (cleaned.find("{"), cleaned.find("[")) if position >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        closing = "}" if cleaned[start] == "{" else "]"
        end = cleaned.rfind(closing)
        if end < start:
            raise
        return json.loads(cleaned[start : end + 1])


class OllamaClient:
    def __init__(self, base_url: str, cache_dir: Path, timeout: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.timeout = timeout
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _request(self, endpoint: str, payload: dict | None = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise OllamaUnavailableError(
                f"Ollama generation exceeded the {self.timeout}-second local inference timeout."
            ) from exc
        except (urllib.error.URLError, ConnectionError) as exc:
            raise OllamaUnavailableError(
                "Ollama is not reachable. Install and start Ollama, then pull the configured models."
            ) from exc

    def available_models(self) -> list[str]:
        response = self._request("/api/tags")
        return [item.get("name", "") for item in response.get("models", [])]

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        json_mode: bool = True,
        json_schema: dict | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        num_predict: int = 400,
        num_ctx: int | None = None,
        use_cache: bool = True,
    ) -> GenerationResult:
        json_schema = _remove_text_limits(json_schema) if json_schema is not None else None
        key_material = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "json": json_mode,
                "schema": json_schema,
                "temperature": temperature,
                "seed": seed,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
            sort_keys=True,
        )
        key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{key}.json"
        if use_cache and cache_path.exists():
            cached = read_json(cache_path)
            cached.pop("cached", None)
            return GenerationResult(**cached, cached=True)

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_predict": num_predict,
                **({"num_ctx": num_ctx} if num_ctx is not None else {}),
            },
        }
        if json_schema is not None:
            payload["format"] = json_schema
        elif json_mode:
            payload["format"] = "json"
        started = time.perf_counter()
        response = self._request("/api/generate", payload)
        result = GenerationResult(
            model=model,
            response=response.get("response", ""),
            elapsed_seconds=round(time.perf_counter() - started, 4),
            prompt_eval_count=response.get("prompt_eval_count"),
            eval_count=response.get("eval_count"),
        )
        write_json(cache_path, {**result.__dict__, "cached": False})
        return result
