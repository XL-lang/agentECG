from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from smolagents import Tool


SKILL_SCRIPT_DIR = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "ecgdeli-analysis" / "scripts"
if str(SKILL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPT_DIR))

from ecgdeli_analysis import (  # noqa: E402
    ECGDeliSession,
    get_amplitude_value,
    get_fiducial_point,
    get_height,
    get_interval_value,
    get_wave_bounds,
    prepare_ecgdeli_session,
    slice_wave,
)


_SESSION_REGISTRY: dict[str, ECGDeliSession] = {}


class ECGDeliPrepareTool(Tool):
    name = "ECGDeli_prepare_tool"
    weight = 5
    description = (
        "Prepare a fine-grained ECGdeli session from a samples x leads ECG matrix and sampling rate. "
        "Use this before any ECGDeli measurement or slicing call so later steps can reuse the same "
        "fiducial points, interval features, and amplitude features by session_id."
    )
    inputs = {
        "data": {
            "type": "array",
            "description": "ECG matrix shaped samples x leads.",
        },
        "fs": {
            "type": "number",
            "description": "Sampling frequency in Hz. Must be positive.",
        },
    }
    output_type = "string"

    def forward(self, data: list[list[float]] | np.ndarray, fs: float):  # type: ignore
        session = prepare_ecgdeli_session(data, fs)
        _SESSION_REGISTRY[session.session_id] = session
        return json.dumps(session.to_summary(), ensure_ascii=False)


class ECGDeliMeasurementTool(Tool):
    name = "ECGDeli_measurement_tool"
    weight = 5
    description = (
        "Run a fine-grained ECGdeli measurement from an existing session_id. Supported operations are "
        "`get_fiducial_point`, `get_wave_bounds`, `slice_wave`, `get_interval_value`, "
        "`get_amplitude_value`, and `get_height`. Prefer this tool for beat-level timing, amplitude, "
        "fiducial-point, and wave-boundary evidence after calling ECGDeli_prepare_tool."
    )
    inputs = {
        "session_id": {
            "type": "string",
            "description": "Session id returned by ECGDeli_prepare_tool.",
        },
        "operation": {
            "type": "string",
            "description": "One of get_fiducial_point, get_wave_bounds, slice_wave, get_interval_value, get_amplitude_value, get_height.",
        },
        "beat_index": {
            "type": "integer",
            "description": "0-based beat index for the requested measurement.",
        },
        "lead_index": {
            "type": "integer",
            "description": "0-based lead index when the operation is lead-specific. Use -1 when not needed.",
            "nullable": True,
        },
        "wave_name": {
            "type": "string",
            "description": "Wave name such as p_wave, qrs_complex, t_wave, pr_interval, qt_interval, or st_segment.",
            "nullable": True,
        },
        "feature_name": {
            "type": "string",
            "description": "Feature name such as qrs_duration, qt_interval, rr_interval, p_amplitude, or r_amplitude.",
            "nullable": True,
        },
        "point_name": {
            "type": "string",
            "description": "Fiducial point name such as p_onset, r_peak, qrs_offset, or t_offset.",
            "nullable": True,
        },
        "synchronized": {
            "type": "boolean",
            "description": "Use synchronized multi-lead timing features when true.",
            "nullable": True,
        },
        "relative_to": {
            "type": "string",
            "description": "Reference for height measurement. v1 supports baseline.",
            "nullable": True,
        },
        "include_samples": {
            "type": "boolean",
            "description": "When true, slice_wave returns full samples in addition to the default summary.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(  # type: ignore
        self,
        session_id: str,
        operation: str,
        beat_index: int,
        lead_index: int = -1,
        wave_name: str = "",
        feature_name: str = "",
        point_name: str = "",
        synchronized: bool = False,
        relative_to: str = "baseline",
        include_samples: bool = False,
    ):
        session = _SESSION_REGISTRY.get(session_id)
        if session is None:
            return json.dumps(
                {
                    "status": "error",
                    "valid": False,
                    "error_identifier": "UnknownSessionId",
                    "error_message": f"Unknown ECGDeli session_id: {session_id}",
                },
                ensure_ascii=False,
            )

        lead = None if lead_index < 0 else lead_index
        operation_key = operation.strip()
        if operation_key == "get_fiducial_point":
            result = get_fiducial_point(session, beat_index, point_name, lead)
        elif operation_key == "get_wave_bounds":
            result = get_wave_bounds(session, beat_index, wave_name, lead)
        elif operation_key == "slice_wave":
            if lead is None:
                result = {
                    "status": "error",
                    "valid": False,
                    "error_identifier": "MissingLeadIndex",
                    "error_message": "lead_index is required for slice_wave.",
                }
            else:
                result = slice_wave(session, beat_index, wave_name, lead, include_samples=include_samples)
        elif operation_key == "get_interval_value":
            result = get_interval_value(session, beat_index, feature_name, lead, synchronized)
        elif operation_key == "get_amplitude_value":
            if lead is None:
                result = {
                    "status": "error",
                    "valid": False,
                    "error_identifier": "MissingLeadIndex",
                    "error_message": "lead_index is required for get_amplitude_value.",
                }
            else:
                result = get_amplitude_value(session, beat_index, feature_name, lead)
        elif operation_key == "get_height":
            if lead is None:
                result = {
                    "status": "error",
                    "valid": False,
                    "error_identifier": "MissingLeadIndex",
                    "error_message": "lead_index is required for get_height.",
                }
            else:
                result = get_height(session, beat_index, wave_name, lead, relative_to)
        else:
            result = {
                "status": "error",
                "valid": False,
                "error_identifier": "UnknownOperation",
                "error_message": f"Unsupported ECGDeli operation: {operation}",
            }
        return json.dumps(result, ensure_ascii=False)
