import base64
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from utils import qwen_vl_plus, vlm_client


DEFAULT_FS = 500
DEFAULT_MAX_SECONDS = 3.0
INVALID_SLICE_CONFIDENCE_THRESHOLD = 0.7
SEGMENT_KEYWORDS = {
    "qrs": ["qrs", "qrs complex"],
    "qt": ["qt", "qt interval"],
    "pr": ["pr", "pr interval"],
    "st": ["st", "st segment"],
    "p_wave": ["p_wave", "p wave", "pwave"],
    "t_wave": ["t_wave", "t wave", "twave"],
    "segment": ["segment", "slice", "window"],
}


@dataclass
class VariableTrace:
    name: str
    value: Any
    summary: dict[str, Any]
    score: int


def ecg_variable_to_image(
    data: Any,
    fs: int = DEFAULT_FS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    output_dir: str | None = None,
    variable_name: str | None = None,
    segment_index: str | None = None,
) -> str:
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError("ECG variable image conversion requires a 1D array.")
    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive.")

    max_points = int(fs * max_seconds)
    if arr.size > max_points:
        raise ValueError(
            f"ECG variable length {arr.size} exceeds max allowed {max_points} points "
            f"({max_seconds}s at fs={fs})."
        )

    target_dir = Path(output_dir or tempfile.gettempdir())
    target_dir.mkdir(parents=True, exist_ok=True)
    image_path = target_dir / f"ecg_reflection_{uuid.uuid4().hex}.png"
    time_axis = np.arange(arr.size) / fs

    title_parts = [
        variable_name or "ECG variable",
        f"fs={fs}Hz",
        f"n={arr.size}",
        f"duration={arr.size / fs:.3f}s",
    ]
    if segment_index:
        title_parts.append(f"segment={segment_index}")

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, arr, color="#1f77b4", linewidth=1.2)
    plt.title(" | ".join(title_parts))
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    return str(image_path)


def capture_agent_variables(
    agent: Any,
    data_analysis_memory: list[Any],
    *,
    fs: int = DEFAULT_FS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    max_candidates: int = 5,
) -> dict[str, Any]:
    state = _get_executor_state(agent)
    code_text = "\n".join(_extract_code_texts(data_analysis_memory)).lower()
    summaries: list[dict[str, Any]] = []
    candidates: list[VariableTrace] = []

    for name, value in state.items():
        if _should_skip_variable(name, value):
            continue

        summary = _summarize_variable(name, value, code_text, fs=fs, max_seconds=max_seconds)
        if summary is None:
            continue
        summaries.append(summary)

        if summary.get("is_plot_candidate"):
            candidates.append(
                VariableTrace(
                    name=name,
                    value=value,
                    summary=summary,
                    score=int(summary.get("candidate_score", 0)),
                )
            )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return {
        "summaries": summaries,
        "candidates": candidates[:max_candidates],
        "code_hints": _extract_code_hints(code_text),
    }


class SegmentSliceAuditor:
    def __init__(
        self,
        *,
        fs: int = DEFAULT_FS,
        max_seconds: float = DEFAULT_MAX_SECONDS,
        confidence_threshold: float = INVALID_SLICE_CONFIDENCE_THRESHOLD,
    ):
        self.fs = fs
        self.max_seconds = max_seconds
        self.confidence_threshold = confidence_threshold

    def audit(
        self,
        *,
        question: str,
        choices: list[str],
        variable_capture: dict[str, Any],
        data_analysis_memory: list[Any],
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        if os.environ.get("AGENT_ECG_DISABLE_VLM_SLICE_AUDIT") == "1":
            return self._empty_result("disabled")

        candidates = variable_capture.get("candidates", [])
        if not candidates:
            return self._empty_result("no_eligible_variables")

        checked: list[dict[str, Any]] = []
        for candidate in candidates:
            summary = candidate.summary
            fs = int(summary.get("fs") or self.fs)
            try:
                image_path = ecg_variable_to_image(
                    candidate.value,
                    fs=fs,
                    max_seconds=self.max_seconds,
                    output_dir=output_dir,
                    variable_name=candidate.name,
                    segment_index=str(summary.get("source_hint", "")),
                )
            except Exception as exc:
                checked.append(
                    {
                        "checked_variable": candidate.name,
                        "verdict": "uncertain",
                        "reason": f"image_conversion_failed: {exc}",
                        "confidence": 0.0,
                        "segment_type": summary.get("segment_type", "unknown"),
                        "morphology_assessment": "image conversion failed before morphology audit",
                        "supports_target_feature": None,
                    }
                )
                continue

            try:
                audit = self._ask_vlm(
                    image_path=image_path,
                    question=question,
                    choices=choices,
                    variable_summary=summary,
                    memory_text="\n".join(_extract_code_texts(data_analysis_memory))[:6000],
                )
            except Exception as exc:
                audit = {
                    "checked_variable": candidate.name,
                    "verdict": "uncertain",
                    "reason": f"vlm_audit_failed: {exc}",
                    "confidence": 0.0,
                    "segment_type": summary.get("segment_type", "unknown"),
                    "morphology_assessment": "vlm audit failed before morphology assessment",
                    "supports_target_feature": None,
                }

            checked.append(_normalize_audit(audit, candidate.name, summary))
            if _is_high_confidence_segmentation_error(checked[-1], self.confidence_threshold):
                return {
                    "verdict": "segmentation_error",
                    "reason": checked[-1]["reason"],
                    "checked_variable": checked[-1]["checked_variable"],
                    "segment_type": checked[-1].get("segment_type", "unknown"),
                    "confidence": checked[-1].get("confidence", 0.0),
                    "morphology_assessment": checked[-1].get("morphology_assessment", ""),
                    "supports_target_feature": checked[-1].get("supports_target_feature"),
                    "checked": checked,
                }
            if _is_high_confidence_invalid(checked[-1], self.confidence_threshold):
                return {
                    "verdict": "invalid_slice",
                    "reason": checked[-1]["reason"],
                    "checked_variable": checked[-1]["checked_variable"],
                    "segment_type": checked[-1].get("segment_type", "unknown"),
                    "confidence": checked[-1].get("confidence", 0.0),
                    "morphology_assessment": checked[-1].get("morphology_assessment", ""),
                    "supports_target_feature": checked[-1].get("supports_target_feature"),
                    "checked": checked,
                }

        valid_count = sum(1 for item in checked if item.get("verdict") == "valid")
        verdict = "valid" if valid_count else "uncertain"
        reason = "VLM did not identify a high-confidence segment slicing error."
        if not checked:
            reason = "No variables could be converted and audited."
        return {
            "verdict": verdict,
            "reason": reason,
            "checked_variable": checked[0]["checked_variable"] if checked else "",
            "segment_type": checked[0].get("segment_type", "unknown") if checked else "unknown",
            "confidence": max((float(item.get("confidence", 0.0)) for item in checked), default=0.0),
            "morphology_assessment": checked[0].get("morphology_assessment", "") if checked else "",
            "supports_target_feature": _merge_support_signal(checked),
            "checked": checked,
        }

    def _empty_result(self, reason: str) -> dict[str, Any]:
        return {
            "verdict": "uncertain",
            "reason": reason,
            "checked_variable": "",
            "segment_type": "unknown",
            "confidence": 0.0,
            "morphology_assessment": "",
            "supports_target_feature": None,
            "checked": [],
        }

    def _ask_vlm(
        self,
        *,
        image_path: str,
        question: str,
        choices: list[str],
        variable_summary: dict[str, Any],
        memory_text: str,
    ) -> dict[str, Any]:
        prompt = f"""
You are auditing an ECG code agent's intermediate variable during reflection.
Determine whether the plotted 1D ECG variable has morphology and boundaries that support the target feature.

Question: {question}
Allowed choices: {choices}
Variable summary: {json.dumps(variable_summary, ensure_ascii=False)}
Relevant code snippets:
{memory_text}

Return strict JSON only:
{{
  "verdict": "valid" | "invalid_slice" | "segmentation_error" | "uncertain",
  "reason": "short reason",
  "checked_variable": "{variable_summary.get('name', '')}",
  "segment_type": "{variable_summary.get('segment_type', 'unknown')}",
  "confidence": 0.0,
  "morphology_assessment": "brief morphology/boundary assessment",
  "supports_target_feature": true
}}

Use segmentation_error only when the slice boundaries/object are clearly wrong for the intended target feature,
such as the wrong wave, missing critical onset/offset, or a shifted window that makes the task unsalvageable.
Use invalid_slice when the slice quality is poor, incomplete, or noisy, but you cannot confidently attribute it to a segment-boundary/object error.
Use valid when morphology and boundaries are adequate for the target feature. Use uncertain when evidence is mixed.
"""
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_encode_image(image_path)}"},
            },
            {"type": "text", "text": prompt},
        ]
        completion = vlm_client.chat.completions.create(
            model=qwen_vl_plus["model"],
            messages=[
                {"role": "system", "content": [{"type": "text", "text": "You are a careful ECG segment auditor."}]},
                {"role": "user", "content": content},
            ],
            temperature=0,
        )
        text = completion.choices[0].message.content
        return _parse_json_object(text)


def _get_executor_state(agent: Any) -> dict[str, Any]:
    executor = getattr(agent, "python_executor", None)
    state = getattr(executor, "state", None)
    if isinstance(state, dict):
        return state
    return {}


def _should_skip_variable(name: str, value: Any) -> bool:
    if name.startswith("_") or name in {"final_answer", "retrieved_micro_skills"}:
        return True
    module = getattr(value.__class__, "__module__", "")
    class_name = value.__class__.__name__.lower()
    if "smolagents" in module or "tool" in class_name or class_name == "ecgsignals":
        return True
    return False


def _summarize_variable(
    name: str,
    value: Any,
    code_text: str,
    *,
    fs: int,
    max_seconds: float,
) -> dict[str, Any] | None:
    try:
        arr = np.asarray(value)
    except Exception:
        return _summarize_scalar(name, value, code_text)

    if arr.ndim == 0:
        return _summarize_scalar(name, arr.item(), code_text)

    numeric = np.issubdtype(arr.dtype, np.number)
    length = int(arr.shape[0]) if arr.ndim >= 1 else 0
    source_hint = _source_hint_for_name(name, code_text)
    segment_type = _infer_segment_type(name, source_hint, code_text)
    candidate_score = _candidate_score(name, source_hint, segment_type, arr, numeric, fs, max_seconds)
    summary: dict[str, Any] = {
        "name": name,
        "type": type(value).__name__,
        "shape": list(arr.shape),
        "length": length,
        "dtype": str(arr.dtype),
        "source_hint": source_hint,
        "segment_type": segment_type,
        "is_plot_candidate": candidate_score > 0,
        "candidate_score": candidate_score,
        "fs": fs,
    }
    if numeric and arr.size:
        flat = arr.astype(float).reshape(-1)
        summary["preview_stats"] = {
            "min": float(np.nanmin(flat)),
            "max": float(np.nanmax(flat)),
            "mean": float(np.nanmean(flat)),
            "std": float(np.nanstd(flat)),
        }
    return summary


def _summarize_scalar(name: str, value: Any, code_text: str) -> dict[str, Any] | None:
    if isinstance(value, (str, int, float, bool)):
        return {
            "name": name,
            "type": type(value).__name__,
            "shape": [],
            "length": 1,
            "dtype": type(value).__name__,
            "source_hint": _source_hint_for_name(name, code_text),
            "segment_type": _infer_segment_type(name, "", code_text),
            "is_plot_candidate": False,
            "candidate_score": 0,
        }
    return None


def _candidate_score(
    name: str,
    source_hint: str,
    segment_type: str,
    arr: np.ndarray,
    numeric: bool,
    fs: int,
    max_seconds: float,
) -> int:
    if not numeric or arr.ndim != 1 or arr.size == 0:
        return 0
    if arr.size > int(fs * max_seconds):
        return 0

    text = f"{name} {source_hint} {segment_type}".lower()
    score = 1
    if any(token in text for values in SEGMENT_KEYWORDS.values() for token in values):
        score += 3
    if "get_lead_segment" in source_hint or "slice_expression" in source_hint:
        score += 3
    if segment_type != "unknown":
        score += 2
    return score


def _source_hint_for_name(name: str, code_text: str) -> str:
    escaped = re.escape(name.lower())
    hints: list[str] = []
    if re.search(rf"{escaped}\s*=.*get_lead_segment\s*\(", code_text):
        hints.append("assigned_from_get_lead_segment")
    if re.search(rf"{escaped}\s*=.*get_lead_signals\s*\(", code_text):
        hints.append("assigned_from_get_lead_signals")
    if re.search(rf"{escaped}\s*=.*\[.*:.*\]", code_text):
        hints.append("slice_expression")
    if any(token in name.lower() for token in ["segment", "slice", "window"]):
        hints.append("segment_named_variable")
    return ";".join(hints) if hints else "unknown"


def _infer_segment_type(name: str, source_hint: str, code_text: str) -> str:
    text = f"{name} {source_hint} {code_text[:1000]}".lower()
    for segment_type, keywords in SEGMENT_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return segment_type
    return "unknown"


def _extract_code_texts(node: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(node, dict):
        function = node.get("function")
        if isinstance(function, dict) and isinstance(function.get("arguments"), str):
            texts.append(function["arguments"])
        for value in node.values():
            texts.extend(_extract_code_texts(value))
    elif isinstance(node, list):
        for item in node:
            texts.extend(_extract_code_texts(item))
    return texts


def _extract_code_hints(code_text: str) -> dict[str, Any]:
    return {
        "get_lead_segment_calls": re.findall(r"get_lead_segment\s*\(([^)]*)\)", code_text),
        "get_lead_signals_calls": re.findall(r"get_lead_signals\s*\(([^)]*)\)", code_text),
        "slice_assignments": re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=.*?\[.*?:.*?\]", code_text),
    }


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _parse_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("VLM response content is not text.")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object found in VLM response: {text[:200]}")
    return json.loads(match.group(0))


def _normalize_audit(audit: dict[str, Any], candidate_name: str, summary: dict[str, Any]) -> dict[str, Any]:
    verdict = str(audit.get("verdict", "uncertain")).strip()
    if verdict not in {"valid", "invalid_slice", "segmentation_error", "uncertain"}:
        verdict = "uncertain"
    try:
        confidence = float(audit.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    supports_target_feature = audit.get("supports_target_feature")
    if isinstance(supports_target_feature, str):
        lowered = supports_target_feature.strip().lower()
        if lowered in {"true", "yes", "supported"}:
            supports_target_feature = True
        elif lowered in {"false", "no", "unsupported"}:
            supports_target_feature = False
        else:
            supports_target_feature = None
    elif not isinstance(supports_target_feature, bool):
        supports_target_feature = None
    return {
        "checked_variable": str(audit.get("checked_variable") or candidate_name),
        "verdict": verdict,
        "reason": str(audit.get("reason", "")),
        "segment_type": str(audit.get("segment_type") or summary.get("segment_type", "unknown")),
        "confidence": confidence,
        "morphology_assessment": str(audit.get("morphology_assessment", "")),
        "supports_target_feature": supports_target_feature,
    }


def _is_high_confidence_invalid(audit: dict[str, Any], threshold: float) -> bool:
    return audit.get("verdict") == "invalid_slice" and float(audit.get("confidence", 0.0)) >= threshold


def _is_high_confidence_segmentation_error(audit: dict[str, Any], threshold: float) -> bool:
    return audit.get("verdict") == "segmentation_error" and float(audit.get("confidence", 0.0)) >= threshold


def _merge_support_signal(checked: list[dict[str, Any]]) -> bool | None:
    for item in checked:
        value = item.get("supports_target_feature")
        if isinstance(value, bool):
            return value
    return None
