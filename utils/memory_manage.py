from __future__ import annotations

from typing import Any

import numpy as np

try:
    from smolagents.agents import ActionStep, CodeAgent
except ImportError:  # pragma: no cover - local test fallback
    class ActionStep:  # type: ignore[override]
        pass

    class CodeAgent:  # type: ignore[override]
        pass


def process_string(text: str) -> str:
    import re

    def format_list(lst):
        formatted = []
        for i, item in enumerate(lst):
            if i == 10:
                break
            try:
                val = float(item)
                formatted.append(f"{val:.1f}")
            except ValueError:
                formatted.append(item)
            if i == 9 and len(lst) > 10:
                formatted[-1] += "..."
        return f"[{','.join(formatted)}]"

    matches = re.findall(r"\[[^\]]*\]", text)
    result = text
    for match in matches:
        content = match.strip("[]").split(",")
        new_list = format_list([item.strip() for item in content])
        if new_list:
            result = result.replace(match, new_list, 1)
    return result


def summarize_array_like(
    value: Any,
    *,
    max_preview: int = 10,
    length_threshold: int = 32,
):
    try:
        arr = np.asarray(value)
    except Exception:
        return value

    if arr.ndim == 0:
        return value

    if arr.ndim == 1:
        should_summarize = int(arr.shape[0]) > length_threshold
        length = int(arr.shape[0])
    else:
        should_summarize = int(arr.size) > length_threshold
        length = int(arr.shape[0]) if arr.shape else int(arr.size)

    if not should_summarize:
        return value

    flat = arr.reshape(-1)
    preview = [_normalize_scalar(item) for item in flat[:max_preview].tolist()]
    summary: dict[str, Any] = {
        "__type__": "array_summary",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "length": length,
        "size": int(arr.size),
        "preview": preview,
        "preview_limit": max_preview,
        "truncated": True,
        "stats": None,
    }

    if np.issubdtype(arr.dtype, np.number) and arr.size:
        numeric = arr.astype(float, copy=False).reshape(-1)
        summary["stats"] = {
            "min": float(np.nanmin(numeric)),
            "max": float(np.nanmax(numeric)),
            "mean": float(np.nanmean(numeric)),
            "std": float(np.nanstd(numeric)),
        }

    return summary


def summarize_nested_payload(
    value: Any,
    *,
    max_preview: int = 10,
    length_threshold: int = 32,
):
    if isinstance(value, dict):
        return {
            key: summarize_nested_payload(item, max_preview=max_preview, length_threshold=length_threshold)
            for key, item in value.items()
        }
    if isinstance(value, list):
        summarized = summarize_array_like(value, max_preview=max_preview, length_threshold=length_threshold)
        if summarized is not value:
            return summarized
        return [
            summarize_nested_payload(item, max_preview=max_preview, length_threshold=length_threshold)
            for item in value
        ]
    if isinstance(value, tuple):
        summarized = summarize_array_like(value, max_preview=max_preview, length_threshold=length_threshold)
        if summarized is not value:
            return summarized
        return tuple(
            summarize_nested_payload(item, max_preview=max_preview, length_threshold=length_threshold)
            for item in value
        )
    return summarize_array_like(value, max_preview=max_preview, length_threshold=length_threshold)


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sanitize_action_step(memory_step: ActionStep) -> None:
    for attr_name, attr_value in vars(memory_step).items():
        if attr_name == "error":
            continue
        try:
            sanitized = summarize_nested_payload(attr_value)
        except Exception:
            continue
        if sanitized is not attr_value:
            setattr(memory_step, attr_name, sanitized)


def clean_memory(memory_step: ActionStep, agent: CodeAgent, log=True) -> None:
    del log
    if not isinstance(memory_step, ActionStep):
        return

    last_step = memory_step
    error = getattr(last_step, "error", None)
    if error:
        error.message = process_string(error.message)
    else:
        _sanitize_action_step(last_step)

    cleaned_steps = []
    for step in getattr(getattr(agent, "memory", None), "steps", []):
        if isinstance(step, ActionStep) and getattr(step, "error", None):
            continue
        cleaned_steps.append(step)

    if hasattr(agent, "memory") and hasattr(agent.memory, "steps"):
        agent.memory.steps = cleaned_steps
