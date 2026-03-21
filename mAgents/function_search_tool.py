import json
import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from smolagents import Tool

from utils import reason_client


logger = logging.getLogger(__name__)

_DEFAULT_QDRANT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_PREFERRED_PAYLOAD_KEYS = (
    "document",
    "text",
    "content",
    "chunk",
    "body",
    "passage",
    "summary",
    "description",
)


@dataclass(frozen=True)
class _QdrantConfig:
    backend: str
    url: str
    api_key: str | None
    collection: str
    top_k: int
    query_model: str
    timeout_seconds: float


def _normalize_url(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().rstrip("/")
    if not normalized:
        return None
    if "://" not in normalized:
        normalized = f"http://{normalized}"
    return normalized


def _build_qdrant_url() -> str | None:
    explicit_url = _normalize_url(os.getenv("QDRANT_URL"))
    if explicit_url:
        return explicit_url

    host = (os.getenv("QDRANT_HOST") or "").strip()
    if not host:
        return None

    port = (os.getenv("QDRANT_PORT") or "6333").strip() or "6333"
    return _normalize_url(f"{host}:{port}")


def _safe_positive_int(raw_value: str | None, default: int) -> int:
    try:
        parsed = int((raw_value or "").strip())
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _safe_positive_float(raw_value: str | None, default: float) -> float:
    try:
        parsed = float((raw_value or "").strip())
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _load_kb_config() -> _QdrantConfig | None:
    backend = (os.getenv("SEARCH_KB_BACKEND") or "qdrant").strip().lower()
    if backend != "qdrant":
        return None

    url = _build_qdrant_url()
    collection = (os.getenv("QDRANT_COLLECTION") or "").strip()
    if not url or not collection:
        return None

    return _QdrantConfig(
        backend=backend,
        url=url,
        api_key=(os.getenv("QDRANT_API_KEY") or "").strip() or None,
        collection=collection,
        top_k=_safe_positive_int(os.getenv("QDRANT_TOP_K"), 5),
        query_model=(os.getenv("QDRANT_QUERY_MODEL") or _DEFAULT_QDRANT_MODEL).strip() or _DEFAULT_QDRANT_MODEL,
        timeout_seconds=_safe_positive_float(os.getenv("SEARCH_KB_TIMEOUT_SECONDS"), 2.0),
    )


def _probe_qdrant(config: _QdrantConfig) -> bool:
    headers = {}
    if config.api_key:
        headers["api-key"] = config.api_key

    for endpoint in ("/readyz", "/healthz"):
        target = f"{config.url}{endpoint}"
        try:
            req = request.Request(target, headers=headers, method="GET")
            with request.urlopen(req, timeout=config.timeout_seconds) as response:
                if response.status == 200:
                    return True
        except (error.URLError, error.HTTPError, TimeoutError) as exc:
            logger.debug("Qdrant probe failed for %s: %s", target, exc)
    return False


def _get_qdrant_client_components():
    from qdrant_client import QdrantClient, models

    return QdrantClient, models


def _extract_payload_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, (int, float, bool)):
        return str(payload)
    if isinstance(payload, dict):
        for key in _PREFERRED_PAYLOAD_KEYS:
            text = _extract_payload_text(payload.get(key))
            if text:
                return text
        for value in payload.values():
            text = _extract_payload_text(value)
            if text:
                return text
        return ""
    if isinstance(payload, (list, tuple)):
        parts = [_extract_payload_text(item) for item in payload]
        return " ".join(part for part in parts if part).strip()
    return ""


def _build_retrieved_context(points: list[Any]) -> str:
    snippets: list[str] = []
    for index, point in enumerate(points, start=1):
        payload = getattr(point, "payload", None)
        snippet = _extract_payload_text(payload)
        if not snippet and payload is not None:
            try:
                snippet = json.dumps(payload, ensure_ascii=False)
            except (TypeError, ValueError):
                snippet = str(payload)
        if snippet:
            snippets.append(f"[{index}] {snippet}")
    return "\n".join(snippets)


def _query_web_search(task: str) -> str:
    completion = reason_client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f" {task}. "},
        ],
        extra_body={"enable_search": True},
    )
    return completion.choices[0].message.content


def _query_qdrant(task: str, config: _QdrantConfig) -> str | None:
    try:
        QdrantClient, models = _get_qdrant_client_components()
    except ImportError as exc:
        logger.info("Qdrant client unavailable, falling back to web search: %s", exc)
        return None

    client = QdrantClient(
        url=config.url,
        api_key=config.api_key,
        timeout=config.timeout_seconds,
    )
    response = client.query_points(
        collection_name=config.collection,
        query=models.Document(text=task.strip(), model=config.query_model),
        limit=config.top_k,
        with_payload=True,
    )
    points = list(getattr(response, "points", []) or [])
    if not points:
        return None

    retrieved_context = _build_retrieved_context(points)
    if not retrieved_context:
        return None

    completion = reason_client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer only from the provided knowledge base snippets. "
                    "If the snippets are insufficient, say so briefly instead of inventing facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{task.strip()}\n\n"
                    f"Knowledge base snippets:\n{retrieved_context}"
                ),
            },
        ],
    )
    return completion.choices[0].message.content


class SearchTool(Tool):
    name = "search_tool"
    description = (
        "This tool uses large models to perform searches. For ECG tasks with retrieved reusable "
        "micro-skills, use it only for medical ECG knowledge such as feature definitions, normal "
        "ranges, diagnostic criteria, and clinical interpretation rules; do not request Python code "
        "or package implementation examples in that case."
    )
    inputs = {
        "task": {
            "type": "string",
            "description": "The task you want to complete.",
        }
    }
    output_type = "string"

    def forward(self, task: str):
        config = _load_kb_config()
        if config and _probe_qdrant(config):
            try:
                qdrant_answer = _query_qdrant(task, config)
            except Exception as exc:
                logger.warning("Qdrant query failed, falling back to web search: %s", exc)
                qdrant_answer = None
            if qdrant_answer:
                return f"[source=qdrant]\n{qdrant_answer}"

        return f"[source=web_search]\n{_query_web_search(task)}"


search_tool = SearchTool()


if __name__ == "__main__":
    content = "How to use Python code to find the major peaks in a sequence?"
    print(search_tool.forward(task=content))
