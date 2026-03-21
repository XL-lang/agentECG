#!/usr/bin/env python3
"""Fine-grained Python entry point for running ECGdeli measurement primitives."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_TIMEOUT_SECONDS = 300
TIMING_LEADWISE_NAMES = [
    "p_duration",
    "qrs_duration",
    "t_duration",
    "pq_interval",
    "pr_interval",
    "qt_interval",
    "rr_interval",
]
TIMING_SYNC_NAMES = TIMING_LEADWISE_NAMES[:-1] + ["qtc_interval", "rr_interval"]
AMPLITUDE_FEATURE_NAMES = [
    "p_amplitude",
    "q_amplitude",
    "r_amplitude",
    "s_amplitude",
    "t_amplitude",
]
FIDUCIAL_POINT_NAMES = [
    "p_onset",
    "p_peak",
    "p_offset",
    "qrs_onset",
    "q_peak",
    "r_peak",
    "s_peak",
    "qrs_offset",
    "j_point",
    "t_onset",
    "t_peak",
    "t_offset",
    "class_id",
]
POINT_TO_INDEX = {name: index for index, name in enumerate(FIDUCIAL_POINT_NAMES)}
WAVE_BOUND_POINT_NAMES = {
    "p_wave": ("p_onset", "p_offset"),
    "qrs_complex": ("qrs_onset", "qrs_offset"),
    "t_wave": ("t_onset", "t_offset"),
    "pr_interval": ("p_onset", "r_peak"),
    "pq_interval": ("p_onset", "qrs_onset"),
    "qt_interval": ("qrs_onset", "t_offset"),
    "st_segment": ("j_point", "t_onset"),
}
WAVE_HEIGHT_FEATURES = {
    "p_wave": "p_amplitude",
    "q_wave": "q_amplitude",
    "r_wave": "r_amplitude",
    "s_wave": "s_amplitude",
    "t_wave": "t_amplitude",
}

SCRIPT_DIR = Path(__file__).resolve().parent
MATLAB_SCRIPT = SCRIPT_DIR / "run_ecgdeli_analysis.m"
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_ECGDELI_ROOT = REPO_ROOT / "external" / "ECGdeli"
UTILS_DIR = REPO_ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from memory_manage import summarize_array_like


@dataclass
class ECGDeliSession:
    session_id: str
    signal: np.ndarray
    filtered_signal: np.ndarray
    fs: float
    work_dir: str
    results_mat: str
    summary_json: str
    fpt_multichannel: np.ndarray
    fpt_by_lead: list[np.ndarray]
    timing_features: np.ndarray
    timing_features_sync: np.ndarray
    amplitude_features: np.ndarray
    summary: dict[str, Any]

    @property
    def samples(self) -> int:
        return int(self.signal.shape[0])

    @property
    def leads(self) -> int:
        return int(self.signal.shape[1])

    @property
    def beats(self) -> int:
        return int(self.fpt_multichannel.shape[0])

    def to_summary(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "session_id": self.session_id,
            "samples": self.samples,
            "leads": self.leads,
            "beats": self.beats,
            "results_mat": self.results_mat,
            "summary_json": self.summary_json,
            "valid": True,
        }


def prepare_ecgdeli_session(
    signal: Sequence[Sequence[float]] | Any,
    fs: float,
    output_dir: str | os.PathLike[str] | None = None,
    matlab_bin: str = "matlab",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> ECGDeliSession:
    rows = _coerce_signal_matrix(signal)
    signal_array = np.asarray(rows, dtype=float)
    work_dir = _prepare_output_dir(output_dir)
    input_csv = work_dir / "input.csv"
    _write_csv(input_csv, rows)
    return prepare_ecgdeli_file_session(
        input_csv=input_csv,
        fs=fs,
        output_dir=work_dir,
        matlab_bin=matlab_bin,
        timeout=timeout,
        signal_override=signal_array,
    )


def prepare_ecgdeli_file_session(
    input_csv: str | os.PathLike[str],
    fs: float,
    output_dir: str | os.PathLike[str] | None = None,
    matlab_bin: str = "matlab",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    signal_override: np.ndarray | None = None,
) -> ECGDeliSession:
    input_path = Path(input_csv).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    fs_value = _coerce_fs(fs)
    work_dir = _prepare_output_dir(output_dir)
    summary_path = work_dir / "summary.json"
    results_path = work_dir / "results.mat"
    timing_json_path = work_dir / "timing_features.json"
    amplitude_json_path = work_dir / "amplitude_features.json"

    try:
        matlab_path = _resolve_executable(matlab_bin)
        _validate_driver_files()
        ecgdeli_root = _resolve_ecgdeli_root()
    except Exception as exc:
        raise RuntimeError(
            json.dumps(
                _error_payload(
                    error_identifier=type(exc).__name__,
                    error_message=str(exc),
                    results_path=results_path,
                    summary_path=summary_path,
                    timing_json_path=timing_json_path,
                    amplitude_json_path=amplitude_json_path,
                ),
                ensure_ascii=False,
            )
        ) from exc

    command = _matlab_batch_command(input_path, fs_value, work_dir)
    env = os.environ.copy()
    env.setdefault("ECGDELI_ROOT", str(ecgdeli_root))

    try:
        completed = subprocess.run(
            [matlab_path, "-batch", command],
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            json.dumps(
                _error_payload(
                    error_identifier="MATLABTimeout",
                    error_message=f"ECGdeli MATLAB execution timed out after {timeout} seconds.",
                    results_path=results_path,
                    summary_path=summary_path,
                    timing_json_path=timing_json_path,
                    amplitude_json_path=amplitude_json_path,
                    extras={
                        "matlab_stdout_tail": _tail(exc.stdout),
                        "matlab_stderr_tail": _tail(exc.stderr),
                    },
                ),
                ensure_ascii=False,
            )
        ) from exc

    summary = _read_json(summary_path)
    if completed.returncode != 0:
        error_payload = summary or _error_payload(
            error_identifier="MATLABExecutionFailed",
            error_message="MATLAB completed with a non-zero exit code and did not produce summary.json.",
            results_path=results_path,
            summary_path=summary_path,
            timing_json_path=timing_json_path,
            amplitude_json_path=amplitude_json_path,
        )
        error_payload["matlab_returncode"] = completed.returncode
        error_payload["matlab_stdout_tail"] = _tail(completed.stdout)
        error_payload["matlab_stderr_tail"] = _tail(completed.stderr)
        raise RuntimeError(json.dumps(error_payload, ensure_ascii=False))

    if not summary or summary.get("status") != "ok":
        raise RuntimeError(
            json.dumps(
                summary
                or _error_payload(
                    error_identifier="MissingSummary",
                    error_message="MATLAB completed without producing a valid summary.json.",
                    results_path=results_path,
                    summary_path=summary_path,
                    timing_json_path=timing_json_path,
                    amplitude_json_path=amplitude_json_path,
                ),
                ensure_ascii=False,
            )
        )

    signal_array = signal_override if signal_override is not None else np.asarray(
        _read_csv_matrix(input_path), dtype=float
    )
    return _build_session(
        signal=signal_array,
        fs=fs_value,
        results_mat=results_path,
        summary=summary,
        work_dir=work_dir,
    )


def get_fiducial_point(
    session: ECGDeliSession,
    beat_index: int,
    point_name: str,
    lead_index: int | None = None,
) -> dict[str, Any]:
    point_key = point_name.strip().lower()
    point_position = POINT_TO_INDEX.get(point_key)
    if point_position is None:
        return _error_measurement(
            feature_name=point_name,
            beat_index=beat_index,
            lead_index=lead_index,
            error_identifier="UnknownPointName",
            error_message=f"Unsupported point_name: {point_name}",
            source="ecgdeli.fiducial_points",
        )

    beat_error = _validate_beat_index(session, beat_index)
    if beat_error is not None:
        return beat_error

    fpt_table = _select_fpt_table(session, lead_index)
    if isinstance(fpt_table, dict):
        return fpt_table

    point_value = int(fpt_table[beat_index, point_position])
    return {
        "status": "ok",
        "value": point_value,
        "units": "sample_index",
        "lead_index": lead_index,
        "beat_index": beat_index,
        "feature_name": point_key,
        "source": "ecgdeli.fiducial_points",
        "valid": point_value > 0,
    }


def get_wave_bounds(
    session: ECGDeliSession,
    beat_index: int,
    wave_name: str,
    lead_index: int | None = None,
) -> dict[str, Any]:
    wave_key = wave_name.strip().lower()
    point_names = WAVE_BOUND_POINT_NAMES.get(wave_key)
    if point_names is None:
        return _error_bounds(
            wave_name=wave_name,
            beat_index=beat_index,
            lead_index=lead_index,
            error_identifier="UnknownWaveName",
            error_message=f"Unsupported wave_name: {wave_name}",
        )

    beat_error = _validate_beat_index(session, beat_index)
    if beat_error is not None:
        return _bounds_from_error(beat_error, wave_name)

    start_result = get_fiducial_point(session, beat_index, point_names[0], lead_index)
    if start_result.get("status") != "ok":
        return _bounds_from_error(start_result, wave_name)
    end_result = get_fiducial_point(session, beat_index, point_names[1], lead_index)
    if end_result.get("status") != "ok":
        return _bounds_from_error(end_result, wave_name)

    start_index = int(start_result["value"])
    end_index = int(end_result["value"])
    if start_index <= 0 or end_index <= 0 or end_index <= start_index:
        return {
            "status": "error",
            "start_index": start_index,
            "end_index": end_index,
            "length_samples": 0,
            "length_ms": 0.0,
            "lead_index": lead_index,
            "beat_index": beat_index,
            "wave_name": wave_key,
            "valid": False,
            "error_identifier": "InvalidWaveBounds",
            "error_message": f"Wave bounds are invalid for {wave_key} at beat {beat_index}.",
        }

    length_samples = end_index - start_index
    return {
        "status": "ok",
        "start_index": start_index,
        "end_index": end_index,
        "length_samples": length_samples,
        "length_ms": _samples_to_ms(length_samples, session.fs),
        "lead_index": lead_index,
        "beat_index": beat_index,
        "wave_name": wave_key,
        "valid": True,
    }


def slice_wave(
    session: ECGDeliSession,
    beat_index: int,
    wave_name: str,
    lead_index: int,
    include_samples: bool = False,
) -> dict[str, Any]:
    bounds = get_wave_bounds(session, beat_index, wave_name, lead_index)
    if bounds.get("status") != "ok":
        return {
            "status": bounds.get("status", "error"),
            "samples_summary": None,
            "start_index": bounds.get("start_index"),
            "end_index": bounds.get("end_index"),
            "length_samples": bounds.get("length_samples", 0),
            "length_ms": bounds.get("length_ms", 0.0),
            "lead_index": lead_index,
            "beat_index": beat_index,
            "wave_name": wave_name,
            "valid": False,
            "error_identifier": bounds.get("error_identifier"),
            "error_message": bounds.get("error_message"),
        }

    signal_error = _validate_lead_index(session, lead_index)
    if signal_error is not None:
        return {
            "status": "error",
            "samples_summary": None,
            "start_index": None,
            "end_index": None,
            "length_samples": 0,
            "length_ms": 0.0,
            "lead_index": lead_index,
            "beat_index": beat_index,
            "wave_name": wave_name,
            "valid": False,
            "error_identifier": signal_error["error_identifier"],
            "error_message": signal_error["error_message"],
        }

    start_index = int(bounds["start_index"])
    end_index = int(bounds["end_index"])
    samples = session.filtered_signal[start_index:end_index, lead_index].astype(float).tolist()
    result = {
        "status": "ok",
        "start_index": start_index,
        "end_index": end_index,
        "length_samples": len(samples),
        "length_ms": _samples_to_ms(len(samples), session.fs),
        "lead_index": lead_index,
        "beat_index": beat_index,
        "wave_name": wave_name,
        "valid": True,
        "samples_summary": summarize_array_like(samples, length_threshold=0),
    }
    if include_samples:
        result["samples"] = samples
    return result


def get_interval_value(
    session: ECGDeliSession,
    beat_index: int,
    feature_name: str,
    lead_index: int | None = None,
    synchronized: bool = False,
) -> dict[str, Any]:
    feature_key = feature_name.strip().lower()
    if synchronized:
        feature_names = TIMING_SYNC_NAMES
        data = session.timing_features_sync
        if feature_key not in feature_names:
            return _error_measurement(
                feature_name=feature_name,
                beat_index=beat_index,
                lead_index=None,
                error_identifier="UnknownFeatureName",
                error_message=f"Unsupported synchronized timing feature: {feature_name}",
                source="ecgdeli.timing_features_sync",
            )
        beat_error = _validate_beat_index(session, beat_index)
        if beat_error is not None:
            return beat_error
        value = float(data[beat_index, feature_names.index(feature_key)])
        return _measurement_result(
            value=value,
            units="ms",
            lead_index=None,
            beat_index=beat_index,
            feature_name=feature_key,
            source="ecgdeli.timing_features_sync",
        )

    if lead_index is None:
        return _error_measurement(
            feature_name=feature_name,
            beat_index=beat_index,
            lead_index=None,
            error_identifier="MissingLeadIndex",
            error_message="lead_index is required for leadwise interval measurements.",
            source="ecgdeli.timing_features",
        )
    lead_error = _validate_lead_index(session, lead_index)
    if lead_error is not None:
        return lead_error
    beat_error = _validate_beat_index(session, beat_index)
    if beat_error is not None:
        return beat_error
    if feature_key not in TIMING_LEADWISE_NAMES:
        return _error_measurement(
            feature_name=feature_name,
            beat_index=beat_index,
            lead_index=lead_index,
            error_identifier="UnknownFeatureName",
            error_message=f"Unsupported leadwise timing feature: {feature_name}",
            source="ecgdeli.timing_features",
        )
    value = float(session.timing_features[lead_index, beat_index, TIMING_LEADWISE_NAMES.index(feature_key)])
    return _measurement_result(
        value=value,
        units="ms",
        lead_index=lead_index,
        beat_index=beat_index,
        feature_name=feature_key,
        source="ecgdeli.timing_features",
    )


def get_amplitude_value(
    session: ECGDeliSession,
    beat_index: int,
    feature_name: str,
    lead_index: int,
) -> dict[str, Any]:
    feature_key = feature_name.strip().lower()
    lead_error = _validate_lead_index(session, lead_index)
    if lead_error is not None:
        return lead_error
    beat_error = _validate_beat_index(session, beat_index)
    if beat_error is not None:
        return beat_error
    if feature_key not in AMPLITUDE_FEATURE_NAMES:
        return _error_measurement(
            feature_name=feature_name,
            beat_index=beat_index,
            lead_index=lead_index,
            error_identifier="UnknownFeatureName",
            error_message=f"Unsupported amplitude feature: {feature_name}",
            source="ecgdeli.amplitude_features",
        )
    value = float(session.amplitude_features[lead_index, beat_index, AMPLITUDE_FEATURE_NAMES.index(feature_key)])
    return _measurement_result(
        value=value,
        units="signal_units",
        lead_index=lead_index,
        beat_index=beat_index,
        feature_name=feature_key,
        source="ecgdeli.amplitude_features",
    )


def get_height(
    session: ECGDeliSession,
    beat_index: int,
    wave_name: str,
    lead_index: int,
    relative_to: str = "baseline",
) -> dict[str, Any]:
    if relative_to.strip().lower() != "baseline":
        return _error_measurement(
            feature_name=f"{wave_name}_height",
            beat_index=beat_index,
            lead_index=lead_index,
            error_identifier="UnsupportedReference",
            error_message=f"Unsupported relative_to value: {relative_to}",
            source="ecgdeli.filtered_signal",
        )

    wave_key = wave_name.strip().lower()
    amplitude_feature = WAVE_HEIGHT_FEATURES.get(wave_key)
    if amplitude_feature is not None:
        result = get_amplitude_value(session, beat_index, amplitude_feature, lead_index)
        if result.get("status") != "ok":
            return result
        result["feature_name"] = f"{wave_key}_height"
        result["source"] = "ecgdeli.amplitude_features"
        return result

    if wave_key == "qrs_complex":
        slice_result = slice_wave(session, beat_index, wave_key, lead_index, include_samples=True)
        if slice_result.get("status") != "ok":
            return _error_measurement(
                feature_name=f"{wave_key}_height",
                beat_index=beat_index,
                lead_index=lead_index,
                error_identifier=slice_result.get("error_identifier", "SliceFailed"),
                error_message=slice_result.get("error_message", "Unable to slice QRS complex."),
                source="ecgdeli.filtered_signal",
            )
        values = np.asarray(slice_result["samples"], dtype=float)
        value = float(values[np.argmax(np.abs(values))]) if values.size else 0.0
        return _measurement_result(
            value=value,
            units="signal_units",
            lead_index=lead_index,
            beat_index=beat_index,
            feature_name=f"{wave_key}_height",
            source="ecgdeli.filtered_signal",
        )

    return _error_measurement(
        feature_name=f"{wave_name}_height",
        beat_index=beat_index,
        lead_index=lead_index,
        error_identifier="UnknownWaveName",
        error_message=f"Unsupported wave_name for height measurement: {wave_name}",
        source="ecgdeli.filtered_signal",
    )


def measure_qrs_duration(
    session: ECGDeliSession,
    beat_index: int,
    lead_index: int | None = None,
    synchronized: bool = False,
) -> dict[str, Any]:
    return get_interval_value(session, beat_index, "qrs_duration", lead_index, synchronized)


def measure_qt_interval(
    session: ECGDeliSession,
    beat_index: int,
    lead_index: int | None = None,
    synchronized: bool = False,
) -> dict[str, Any]:
    return get_interval_value(session, beat_index, "qt_interval", lead_index, synchronized)


def measure_rr_interval(
    session: ECGDeliSession,
    beat_index: int,
    lead_index: int | None = None,
    synchronized: bool = False,
) -> dict[str, Any]:
    return get_interval_value(session, beat_index, "rr_interval", lead_index, synchronized)


def measure_p_amplitude(
    session: ECGDeliSession,
    beat_index: int,
    lead_index: int,
) -> dict[str, Any]:
    return get_amplitude_value(session, beat_index, "p_amplitude", lead_index)


def load_ecgdeli_results(results_mat: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(results_mat).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"results.mat not found: {path}")

    raw = _load_results_mat(path)
    return {
        "results_mat": str(path),
        "fiducial_points_available": raw["fpt_multichannel"].size > 0,
        "fpt_multichannel_shape": list(raw["fpt_multichannel"].shape),
        "timing_features": {
            "leadwise": _array_payload(raw["timing_features"], TIMING_LEADWISE_NAMES, units="ms"),
            "synchronized": _array_payload(raw["timing_features_sync"], TIMING_SYNC_NAMES, units="ms"),
        },
        "amplitude_features": _array_payload(raw["amplitude_features"], AMPLITUDE_FEATURE_NAMES, units="signal_units"),
    }


def _build_session(
    *,
    signal: np.ndarray,
    fs: float,
    results_mat: Path,
    summary: dict[str, Any],
    work_dir: Path,
) -> ECGDeliSession:
    raw = _load_results_mat(results_mat)
    return ECGDeliSession(
        session_id=uuid.uuid4().hex,
        signal=np.asarray(signal, dtype=float),
        filtered_signal=raw["filtered_signal"],
        fs=float(fs),
        work_dir=str(work_dir),
        results_mat=str(results_mat),
        summary_json=str(Path(summary["summary_json"]).resolve()),
        fpt_multichannel=raw["fpt_multichannel"],
        fpt_by_lead=raw["fpt_by_lead"],
        timing_features=raw["timing_features"],
        timing_features_sync=raw["timing_features_sync"],
        amplitude_features=raw["amplitude_features"],
        summary=summary,
    )


def _load_results_mat(path: Path) -> dict[str, Any]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError("scipy is required to read ECGdeli results.mat files.") from exc

    raw = loadmat(path, squeeze_me=False, struct_as_record=False)
    fpt_multichannel = np.asarray(raw.get("fptMultiChannel"), dtype=int)
    fpt_cell = raw.get("fptCell")
    if fpt_cell is None:
        raise RuntimeError("results.mat is missing fptCell.")
    fpt_by_lead = [np.asarray(fpt_cell[index, 0], dtype=int) for index in range(fpt_cell.shape[0])]

    filtered_signal = raw.get("ecgFiltered")
    if filtered_signal is None:
        raise RuntimeError("results.mat is missing ecgFiltered; rerun ECGdeli with the updated driver.")

    return {
        "fpt_multichannel": fpt_multichannel,
        "fpt_by_lead": fpt_by_lead,
        "timing_features": np.asarray(raw.get("timingFeatures"), dtype=float),
        "timing_features_sync": np.asarray(raw.get("timingFeaturesSync"), dtype=float),
        "amplitude_features": np.asarray(raw.get("amplitudeFeatures"), dtype=float),
        "filtered_signal": np.asarray(filtered_signal, dtype=float),
    }


def _coerce_signal_matrix(signal: Sequence[Sequence[float]] | Any) -> list[list[float]]:
    if hasattr(signal, "tolist"):
        signal = signal.tolist()

    if not isinstance(signal, Sequence) or isinstance(signal, (str, bytes)):
        raise ValueError("signal must be a 2-D numeric matrix shaped samples x leads")

    rows: list[list[float]] = []
    width: int | None = None
    for row in signal:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError("signal must be a 2-D numeric matrix shaped samples x leads")
        numeric_row = [_coerce_number(value) for value in row]
        if width is None:
            width = len(numeric_row)
            if width < 1:
                raise ValueError("signal must have at least one lead")
        elif len(numeric_row) != width:
            raise ValueError("all signal rows must have the same number of leads")
        rows.append(numeric_row)

    if len(rows) < 2:
        raise ValueError("signal must contain at least two samples")
    return rows


def _coerce_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"signal contains a non-numeric value: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"signal contains a non-finite value: {value!r}")
    return number


def _coerce_fs(fs: float) -> float:
    try:
        fs_value = float(fs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fs must be a positive number, got {fs!r}") from exc
    if not math.isfinite(fs_value) or fs_value <= 0:
        raise ValueError(f"fs must be a positive number, got {fs!r}")
    return fs_value


def _prepare_output_dir(output_dir: str | os.PathLike[str] | None) -> Path:
    if output_dir is None:
        return Path(tempfile.mkdtemp(prefix="ecgdeli_analysis_"))
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, rows: Iterable[Iterable[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerows(rows)


def _read_csv_matrix(path: Path) -> list[list[float]]:
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.reader(file_obj)
        return [[float(cell) for cell in row] for row in reader]


def _resolve_executable(executable: str) -> str:
    path = Path(executable).expanduser()
    if path.is_absolute() or len(path.parts) > 1:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        raise FileNotFoundError(f"MATLAB executable not found or not executable: {path}")

    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(
            f"MATLAB executable not found on PATH: {executable}. "
            "Pass matlab_bin='/home/xl/MATLAB/R2025b/bin/matlab' if needed."
        )
    return resolved


def _validate_driver_files() -> None:
    if not MATLAB_SCRIPT.is_file():
        raise FileNotFoundError(f"MATLAB ECGdeli driver not found: {MATLAB_SCRIPT}")


def _resolve_ecgdeli_root() -> Path:
    ecgdeli_root = Path(os.environ.get("ECGDELI_ROOT", DEFAULT_ECGDELI_ROOT)).expanduser().resolve()
    if not ecgdeli_root.is_dir():
        raise FileNotFoundError(
            f"ECGdeli root not found: {ecgdeli_root}. Set ECGDELI_ROOT or restore external/ECGdeli."
        )
    return ecgdeli_root


def _matlab_batch_command(input_csv: Path, fs: float, output_dir: Path) -> str:
    script_dir = _matlab_quote(SCRIPT_DIR)
    input_arg = _matlab_quote(input_csv)
    output_arg = _matlab_quote(output_dir)
    return (
        f"addpath('{script_dir}'); "
        f"run_ecgdeli_analysis('{input_arg}', '{fs:.12g}', '{output_arg}');"
    )


def _matlab_quote(value: str | os.PathLike[str]) -> str:
    return str(value).replace("'", "''")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _array_payload(values: np.ndarray, feature_names: list[str], units: str) -> dict[str, Any]:
    return {
        "feature_names": feature_names,
        "shape": list(values.shape),
        "units": units,
        "values": values.tolist(),
    }


def _measurement_result(
    *,
    value: float,
    units: str,
    lead_index: int | None,
    beat_index: int,
    feature_name: str,
    source: str,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "value": value,
        "units": units,
        "lead_index": lead_index,
        "beat_index": beat_index,
        "feature_name": feature_name,
        "source": source,
        "valid": True,
    }


def _error_measurement(
    *,
    feature_name: str,
    beat_index: int,
    lead_index: int | None,
    error_identifier: str,
    error_message: str,
    source: str,
) -> dict[str, Any]:
    return {
        "status": "error",
        "value": None,
        "units": None,
        "lead_index": lead_index,
        "beat_index": beat_index,
        "feature_name": feature_name,
        "source": source,
        "valid": False,
        "error_identifier": error_identifier,
        "error_message": error_message,
    }


def _error_bounds(
    *,
    wave_name: str,
    beat_index: int,
    lead_index: int | None,
    error_identifier: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "status": "error",
        "start_index": None,
        "end_index": None,
        "length_samples": 0,
        "length_ms": 0.0,
        "lead_index": lead_index,
        "beat_index": beat_index,
        "wave_name": wave_name,
        "valid": False,
        "error_identifier": error_identifier,
        "error_message": error_message,
    }


def _bounds_from_error(error_result: dict[str, Any], wave_name: str) -> dict[str, Any]:
    return {
        "status": "error",
        "start_index": None,
        "end_index": None,
        "length_samples": 0,
        "length_ms": 0.0,
        "lead_index": error_result.get("lead_index"),
        "beat_index": error_result.get("beat_index"),
        "wave_name": wave_name,
        "valid": False,
        "error_identifier": error_result.get("error_identifier"),
        "error_message": error_result.get("error_message"),
    }


def _validate_lead_index(session: ECGDeliSession, lead_index: int) -> dict[str, Any] | None:
    if not isinstance(lead_index, int):
        return _error_measurement(
            feature_name="lead_index",
            beat_index=-1,
            lead_index=lead_index if isinstance(lead_index, int) else None,
            error_identifier="InvalidLeadIndex",
            error_message=f"lead_index must be an int, got {lead_index!r}",
            source="ecgdeli.session",
        )
    if lead_index < 0 or lead_index >= session.leads:
        return _error_measurement(
            feature_name="lead_index",
            beat_index=-1,
            lead_index=lead_index,
            error_identifier="LeadIndexOutOfRange",
            error_message=f"lead_index {lead_index} is out of range for {session.leads} leads.",
            source="ecgdeli.session",
        )
    return None


def _validate_beat_index(session: ECGDeliSession, beat_index: int) -> dict[str, Any] | None:
    if not isinstance(beat_index, int):
        return _error_measurement(
            feature_name="beat_index",
            beat_index=-1,
            lead_index=None,
            error_identifier="InvalidBeatIndex",
            error_message=f"beat_index must be an int, got {beat_index!r}",
            source="ecgdeli.session",
        )
    if beat_index < 0 or beat_index >= session.beats:
        return _error_measurement(
            feature_name="beat_index",
            beat_index=beat_index,
            lead_index=None,
            error_identifier="BeatIndexOutOfRange",
            error_message=f"beat_index {beat_index} is out of range for {session.beats} beats.",
            source="ecgdeli.session",
        )
    return None


def _select_fpt_table(session: ECGDeliSession, lead_index: int | None) -> np.ndarray | dict[str, Any]:
    if lead_index is None:
        return session.fpt_multichannel
    lead_error = _validate_lead_index(session, lead_index)
    if lead_error is not None:
        return lead_error
    return session.fpt_by_lead[lead_index]


def _samples_to_ms(length_samples: int, fs: float) -> float:
    return float(length_samples * 1000.0 / fs)


def _error_payload(
    *,
    error_identifier: str,
    error_message: str,
    results_path: Path,
    summary_path: Path,
    timing_json_path: Path,
    amplitude_json_path: Path,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "status": "error",
        "results_mat": str(results_path),
        "summary_json": str(summary_path),
        "timing_features_json": str(timing_json_path),
        "amplitude_features_json": str(amplitude_json_path),
        "error_identifier": error_identifier,
        "error_message": error_message,
    }
    if extras:
        payload.update(extras)
    return payload


def _tail(text: str | None, limit: int = 4000) -> str:
    if not text:
        return ""
    return text[-limit:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare an ECGdeli session and print its summary.")
    parser.add_argument("--input", required=True, help="CSV ECG matrix shaped samples x leads.")
    parser.add_argument("--fs", required=True, type=float, help="Sampling rate in Hz.")
    parser.add_argument("--output-dir", default=None, help="Directory for ECGdeli artifacts.")
    parser.add_argument("--matlab-bin", default="matlab", help="MATLAB executable path or command.")
    parser.add_argument("--timeout", default=DEFAULT_TIMEOUT_SECONDS, type=int, help="MATLAB timeout in seconds.")
    args = parser.parse_args()

    session = prepare_ecgdeli_file_session(
        args.input,
        args.fs,
        output_dir=args.output_dir,
        matlab_bin=args.matlab_bin,
        timeout=args.timeout,
    )
    print(json.dumps(session.to_summary(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
