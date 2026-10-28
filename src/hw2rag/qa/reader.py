"""Document readers for local generative and deterministic QA baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hw2rag.qa.prompt import EvidenceContext, build_messages


DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class ReaderResponse:
    answer: str
    model: str
    raw_response: str
    duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Reader(Protocol):
    def answer(
        self,
        question: str,
        context: EvidenceContext | None,
    ) -> ReaderResponse: ...


def clean_generated_answer(content: str) -> str:
    value = content.strip()
    if not value:
        return "UNKNOWN"
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(value[start : end + 1])
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
        value = parsed["answer"].strip()
    for marker in ("<tool_call>", "</tool_call>", "<|im_start|>", "<|im_end|>", "<|"):
        if marker in value:
            value = value.split(marker, 1)[0]
    value = re.sub(r"^(?:answer|final answer)\s*:\s*", "", value, flags=re.IGNORECASE)
    value = value.strip().strip("`\"{}").strip()
    value = " ".join(value.split())
    if len(value) > 500:
        value = value[:500].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    return value or "UNKNOWN"


def is_valid_answer(value: str) -> bool:
    lowered = value.casefold()
    if not value.strip() or any(
        marker in lowered for marker in ("<tool_call>", "<|", "|>", "strconv")
    ):
        return False
    return any(character.isalnum() for character in value)


def valid_answer_object(content: str) -> bool:
    try:
        value = json.loads(content)
    except ValueError:
        return False
    return (isinstance(value, dict) and set(value) == {"answer"}
            and isinstance(value["answer"], str) and is_valid_answer(value["answer"]))


class OllamaReader:
    def __init__(
        self,
        *,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 180.0,
        temperature: float = 0.0,
        seed: int = 42,
        num_predict: int = 96,
        num_ctx: int = 8192,
        keep_alive: str = "15m",
        answer_profile: str = "baseline",
        strict_json: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.seed = seed
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        if answer_profile not in {"baseline", "concise", "grounded"}:
            raise ValueError(f"unknown answer profile: {answer_profile}")
        self.answer_profile = answer_profile
        self.strict_json = strict_json or answer_profile == "concise"

    def answer(
        self,
        question: str,
        context: EvidenceContext | None,
        *,
        initial_attempt: dict[str, Any] | None = None,
    ) -> ReaderResponse:
        payload = {
            "model": self.model,
            "messages": build_messages(question, context, answer_profile=self.answer_profile),
            "stream": False,
            "format": ANSWER_SCHEMA,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "seed": self.seed,
                "temperature": self.temperature,
            },
        }
        started = time.monotonic()
        response_data: dict[str, Any] = {}
        content = ""
        answer = "UNKNOWN"
        attempts = 0
        attempt_responses = []
        for attempts in range(1, 3):
            request = Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                if attempts == 1 and initial_attempt is not None:
                    # Only an identity-verified run replay may supply this
                    # cached first attempt; retry messages/options stay exactly
                    # the same as in an ordinary production call.
                    response_data = initial_attempt
                else:
                    with urlopen(request, timeout=self.timeout) as response:
                        response_data = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
            except URLError as exc:
                raise RuntimeError(
                    f"cannot reach Ollama at {self.base_url}; start the local server first"
                ) from exc
            content = str(response_data.get("message", {}).get("content", ""))
            answer = clean_generated_answer(content)
            valid = is_valid_answer(answer)
            if self.strict_json:
                attempt_responses.append({"raw_response": content, "done_reason": response_data.get("done_reason"),
                    "valid_answer_object": valid_answer_object(content), "eval_count": response_data.get("eval_count"),
                    "num_predict": payload["options"]["num_predict"]})
                valid = valid and valid_answer_object(content) and response_data.get("done_reason") != "length"
            if valid:
                break
            if self.strict_json:
                payload["options"]["num_predict"] = max(self.num_predict, 256)
            payload["messages"] = [
                *payload["messages"],
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": "That answer was malformed. Return only valid JSON with a concise answer string.",
                },
            ]
        if not valid:
            answer = "UNKNOWN"
        duration = time.monotonic() - started
        return ReaderResponse(
            answer=answer,
            model=str(response_data.get("model") or self.model),
            raw_response=content,
            duration_seconds=duration,
            metadata={
                "attempts": attempts,
                **({"attempt_responses": attempt_responses, "strict_json_validation": True}
                   if self.strict_json else {}),
                **{
                    key: response_data[key]
                    for key in (
                        "created_at",
                        "done_reason",
                        "eval_count",
                        "eval_duration",
                        "load_duration",
                        "prompt_eval_count",
                        "prompt_eval_duration",
                        "total_duration",
                    )
                    if key in response_data
                },
            },
        )


class ExtractiveReader:
    """A deterministic retrieval-only baseline that emits the top document title."""

    model = "top_document_title_v1"

    def answer(
        self,
        question: str,
        context: EvidenceContext | None,
    ) -> ReaderResponse:
        del question
        started = time.monotonic()
        if context is None or not context.sources:
            answer = "UNKNOWN"
        else:
            answer = str(context.sources[0]["title"]).strip() or "UNKNOWN"
        return ReaderResponse(
            answer=answer,
            model=self.model,
            raw_response=answer,
            duration_seconds=time.monotonic() - started,
        )
