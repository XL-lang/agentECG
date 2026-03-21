from __future__ import annotations

from typing import Any


STRUCTURED_KEYWORDS = {
    "pr": "interval",
    "qrs": "interval",
    "qt": "interval",
    "qtc": "interval",
    "rr": "interval",
    "interval": "interval",
    "duration": "duration",
    "fiducial": "fiducial",
    "beat timing": "beat_timing",
    "timing": "beat_timing",
    "amplitude": "amplitude",
    "voltage": "amplitude",
}

MORPHOLOGY_KEYWORDS = {
    "morphology": "morphology",
    "shape": "morphology",
    "visual pattern": "morphology",
    "visual": "morphology",
    "st change": "morphology",
    "st segment": "morphology",
    "t-wave shape": "morphology",
    "t wave shape": "morphology",
    "noise-sensitive": "noise_sensitive",
    "noise sensitive": "noise_sensitive",
    "noise": "noise_sensitive",
    "artifact": "noise_sensitive",
    "interference": "noise_sensitive",
    "baseline drift": "noise_sensitive",
    "burst noise": "noise_sensitive",
    "static noise": "noise_sensitive",
    "electrode": "noise_sensitive",
    "electrodes problems": "noise_sensitive",
    "contour": "morphology",
}

DURATION_REQUIRED_TAGS = {"interval", "duration", "fiducial", "beat_timing"}


def get_ecgdeli_prerequisites(ecgs: dict[str, Any]) -> dict[str, Any]:
    fs_candidates: list[float] = []
    ecg_count = len(ecgs)

    for ecg in ecgs.values():
        fs = _extract_fs(ecg)
        if fs is not None:
            fs_candidates.append(fs)

    fs_available = bool(fs_candidates)
    fs_consistent = len({round(fs, 6) for fs in fs_candidates}) <= 1 if fs_candidates else False

    return {
        "ecg_count": ecg_count,
        "single_ecg": ecg_count == 1,
        "fs_available": fs_available,
        "fs_consistent": fs_consistent,
        "fs_value": fs_candidates[0] if fs_available and fs_consistent else None,
        "supports_matrix_input": ecg_count == 1 and fs_available and fs_consistent,
    }


def route_ecg_analysis(
    question: str,
    choices: list[str],
    pre_analysis_res: str | None,
    ecgs: dict[str, Any],
) -> dict[str, Any]:
    del choices

    searchable_text = " ".join(
        part for part in [question, pre_analysis_res or ""] if isinstance(part, str)
    ).lower()
    structured_tags = _match_tags(searchable_text, STRUCTURED_KEYWORDS)
    morphology_tags = _match_tags(searchable_text, MORPHOLOGY_KEYWORDS)
    prerequisites = get_ecgdeli_prerequisites(ecgs)

    requires_fs = any(tag in DURATION_REQUIRED_TAGS for tag in structured_tags)
    fs_available = prerequisites["fs_available"] and prerequisites["fs_consistent"]

    preferred_tool = "none"
    fallback_tool = "none"
    reason = "No strong ECG tool routing signal was detected."

    if structured_tags and not morphology_tags:
        preferred_tool = "ECGDeli_measurement_tool"
        fallback_tool = "ECG_fig_anaysis_tool"
        reason = "Question asks for structured interval, timing, fiducial, or amplitude evidence."
    elif morphology_tags and not structured_tags:
        preferred_tool = "ECG_fig_anaysis_tool"
        fallback_tool = "none"
        reason = "Question is morphology-heavy or visually defined."
    elif structured_tags and morphology_tags:
        if any(tag in DURATION_REQUIRED_TAGS for tag in structured_tags):
            preferred_tool = "ECGDeli_measurement_tool"
            fallback_tool = "ECG_fig_anaysis_tool"
            reason = "Question mixes morphology and structured timing evidence; structured measurement takes priority."
        else:
            preferred_tool = "ECG_fig_anaysis_tool"
            fallback_tool = "ECGDeli_measurement_tool"
            reason = "Question mixes morphology and structure, but the main evidence appears visual rather than duration-based."

    if preferred_tool == "ECGDeli_measurement_tool" and not prerequisites["supports_matrix_input"]:
        fallback_tool = "ECG_fig_anaysis_tool"
        if requires_fs and not fs_available:
            reason = "Question needs structured duration evidence, but fs is missing or inconsistent."
        elif not prerequisites["single_ecg"]:
            reason = "Question favors ECGdeli, but multiple ECG objects prevent a single matrix-style ECGdeli call."
        else:
            reason = "Question favors ECGdeli, but its prerequisites are not fully satisfied."

    routing_constraints = _build_constraints(
        preferred_tool=preferred_tool,
        fallback_tool=fallback_tool,
        requires_fs=requires_fs,
        fs_available=fs_available,
        prerequisites=prerequisites,
    )

    task_tags = sorted(structured_tags | morphology_tags)
    return {
        "preferred_tool": preferred_tool,
        "reason": reason,
        "requires_fs": requires_fs,
        "fs_available": fs_available,
        "task_tags": task_tags,
        "fallback_tool": fallback_tool,
        "routing_constraints": routing_constraints,
        "prerequisites": prerequisites,
    }


def compose_routing_context(routing_decision: dict[str, Any]) -> str:
    prerequisites = routing_decision.get("prerequisites", {})
    lines = [
        "ECG tool routing verdict:",
        f"- preferred_tool: {routing_decision.get('preferred_tool', 'none')}",
        f"- reason: {routing_decision.get('reason', '')}",
        f"- fallback_tool: {routing_decision.get('fallback_tool', 'none')}",
        f"- requires_fs: {routing_decision.get('requires_fs', False)}",
        f"- fs_available: {routing_decision.get('fs_available', False)}",
        f"- task_tags: {', '.join(routing_decision.get('task_tags', [])) or 'none'}",
        f"- single_ecg: {prerequisites.get('single_ecg', False)}",
        f"- supports_matrix_input: {prerequisites.get('supports_matrix_input', False)}",
        "Execution constraints:",
        *[f"- {line}" for line in routing_decision.get("routing_constraints", "").splitlines() if line.strip()],
    ]
    return "\n".join(lines)


def _extract_fs(ecg: Any) -> float | None:
    candidates = []
    if hasattr(ecg, "get_fs"):
        candidates.append(getattr(ecg, "get_fs"))
    for attr in ("fs", "sampling_rate"):
        if hasattr(ecg, attr):
            candidates.append(lambda attr_name=attr: getattr(ecg, attr_name))

    for candidate in candidates:
        try:
            value = candidate() if callable(candidate) else candidate
        except Exception:
            continue
        try:
            fs = float(value)
        except (TypeError, ValueError):
            continue
        if fs > 0:
            return fs
    return None


def _match_tags(text: str, keyword_map: dict[str, str]) -> set[str]:
    tags: set[str] = set()
    for phrase, tag in keyword_map.items():
        if phrase in text:
            tags.add(tag)
    return tags


def _build_constraints(
    *,
    preferred_tool: str,
    fallback_tool: str,
    requires_fs: bool,
    fs_available: bool,
    prerequisites: dict[str, Any],
) -> str:
    constraints: list[str] = []

    if preferred_tool == "ECGDeli_measurement_tool":
        constraints.append(
            "If ECGdeli is recommended, call ECGDeli_prepare_tool first, then use ECGDeli_measurement_tool for the smallest required structured measurement before visual inspection."
        )
        constraints.append(
            "Treat wave slicing and segment boundary extraction as ECGdeli responsibilities, not as primary visual-tool responsibilities."
        )
    elif preferred_tool == "ECG_fig_anaysis_tool":
        constraints.append(
            "Use ECG_fig_anaysis_tool first only because the task is currently classified as morphology-heavy."
        )
        constraints.append(
            "Do not use the visual tool as the primary segment slicer when ECGdeli can provide the needed boundaries or wave slice."
        )
        constraints.append(
            "For noise, artifact, or interference classification, prefer visual morphology inspection over ECGdeli beat-boundary measurements."
        )
    else:
        constraints.append(
            "No tool is mandatory; choose the narrowest verifiable evidence path for this task."
        )

    if requires_fs and not fs_available:
        constraints.append(
            "Do not produce duration-based conclusions from hard-coded sampling frequency assumptions."
        )
        constraints.append(
            "If duration evidence is required, state the limitation or fall back to non-duration evidence."
        )

    if preferred_tool == "ECGDeli_measurement_tool" and not prerequisites.get("single_ecg", False):
        constraints.append(
            "Do not improvise multi-ECG matrix packing for ECGdeli; use fallback or answer conservatively."
        )

    if fallback_tool != "none":
        constraints.append(f"If the preferred path fails, fall back to {fallback_tool}.")

    constraints.append(
        "When using ECGdeli, ECGDeli_prepare_tool returns a session summary with a `session_id` field; extract that field once and pass only the raw session_id string to ECGDeli_measurement_tool."
    )
    constraints.append(
        "Do not pass the full prepare-tool payload as `session_id`, and do not index an already extracted session_id string again."
    )

    return "\n".join(constraints)
