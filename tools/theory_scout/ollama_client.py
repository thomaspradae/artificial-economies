from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


@dataclass
class OllamaResult:
    model: str
    content: str
    raw: dict[str, Any]

    @property
    def output_tokens(self) -> int:
        return int(self.raw.get("eval_count") or 0)

    @property
    def output_tokens_per_second(self) -> float | None:
        duration = int(self.raw.get("eval_duration") or 0)
        if duration <= 0:
            return None
        return self.output_tokens / (duration / 1_000_000_000)


class OllamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        *,
        timeout_seconds: int = 240,
        opener: Callable[[urllib.request.Request, int], Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._opener = opener

    def chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        num_predict: int = 900,
        num_ctx: int = 4096,
        num_thread: int | None = None,
        think: bool | None = False,
        keep_alive: int | str = 0,
        format_json: bool = False,
    ) -> OllamaResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
        }
        if think is not None:
            payload["think"] = think
        if format_json:
            payload["format"] = "json"
        if num_thread is not None:
            payload["options"]["num_thread"] = num_thread

        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            if self._opener is not None:
                response = self._opener(request, self.timeout_seconds)
            else:
                response = urllib.request.urlopen(request, timeout=self.timeout_seconds)
            with response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ollama request failed at {self.base_url}: {exc}") from exc
        content = str(data.get("message", {}).get("content") or "")
        return OllamaResult(model=model, content=content, raw=data)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object from model output.

    Local models sometimes wrap the object in markdown fences or short prose.
    This function accepts that, but still fails closed if no parseable object is
    present.
    """

    stripped = text.strip()
    if not stripped:
        raise ValueError("empty model response")
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S)
    candidates = [stripped]
    if fence_match:
        candidates.insert(0, fence_match.group(1))
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(stripped[first : last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("no parseable JSON object in model response")
