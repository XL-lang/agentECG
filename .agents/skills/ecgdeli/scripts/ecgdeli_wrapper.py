#!/usr/bin/env python3
"""Python entry point for running ECGdeli through MATLAB."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_TIMEOUT_SECONDS = 300
SCRIPT_DIR = Path(__file__).resolve().parent
MATLAB_SCRIPT = SCRIPT_DIR / "run_ecgdeli_analysis.m"
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_ECGDELI_ROOT = REPO_ROOT / "external" / "ECGdeli"


def annotate_ecg(
    signal: Sequence[Sequence[float]] | Any,
    fs: float,
    output_dir: str | os.PathLike[str] | None = None,
    matlab_bin: str = "matlab",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run ECGdeli on an in-memory ECG matrix shaped samples x leads."""
    rows = _coerce_signal_matrix(signal)
    work_dir = _prepare_output_dir(output_dir)
    input_csv = work_dir / "input.csv"
    _write_csv(input_csv, rows)
    return annotate_ecg_file(input_csv, fs, work_dir, matlab_bin, timeout)


def annotate_ecg_file(
    input_csv: str | os.PathLike[str],
    fs: float,
    output_dir: str | os.PathLike[str] | None = None,
    matlab_bin: str = "matlab",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run ECGdeli on a CSV ECG matrix shaped samples x leads."""
    input_path = Path(input_csv).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    fs_value = _coerce_fs(fs)
    matlab_path = _resolve_executable(matlab_bin)
    if not MATLAB_SCRIPT.is_file():
        raise FileNotFoundError(f"MATLAB ECGdeli driver not found: {MATLAB_SCRIPT}")

    ecgdeli_root = Path(os.environ.get("ECGDELI_ROOT", DEFAULT_ECGDELI_ROOT)).expanduser().resolve()
    if not ecgdeli_root.is_dir():
        raise FileNotFoundError(
            f"ECGdeli root not found: {ecgdeli_root}. "
            "Set ECGDELI_ROOT or restore external/ECGdeli."
        )

    work_dir = _prepare_output_dir(output_dir)
    summary_path = work_dir / "summary.json"

    command = _matlab_batch_command(input_path, fs_value, work_dir)
    env = os.environ.copy()
    env.setdefault("ECGDELI_ROOT", str(ecgdeli_root))

    completed = subprocess.run(
        [matlab_path, "-batch", command],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(_format_matlab_failure(matlab_path, completed))

    if not summary_path.is_file():
        raise RuntimeError(
            "MATLAB completed without producing summary.json. "
            f"stdout tail:\n{_tail(completed.stdout)}\n"
            f"stderr tail:\n{_tail(completed.stderr)}"
        )

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    if summary.get("status") != "ok":
        raise RuntimeError(f"ECGdeli returned non-ok status: {summary}")

    return summary


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
        path = Path(tempfile.mkdtemp(prefix="ecgdeli_"))
    else:
        path = Path(output_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, rows: Iterable[Iterable[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


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


def _format_matlab_failure(matlab_path: str, completed: subprocess.CompletedProcess[str]) -> str:
    return (
        f"MATLAB ECGdeli execution failed using {matlab_path!r} "
        f"with return code {completed.returncode}.\n"
        f"stdout tail:\n{_tail(completed.stdout)}\n"
        f"stderr tail:\n{_tail(completed.stderr)}"
    )


def _tail(text: str, limit: int = 4000) -> str:
    if not text:
        return ""
    return text[-limit:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ECGdeli through MATLAB.")
    parser.add_argument("--input", required=True, help="CSV ECG matrix shaped samples x leads.")
    parser.add_argument("--fs", required=True, type=float, help="Sampling rate in Hz.")
    parser.add_argument("--output-dir", default=None, help="Directory for summary.json and results.mat.")
    parser.add_argument("--matlab-bin", default="matlab", help="MATLAB executable path or command.")
    parser.add_argument("--timeout", default=DEFAULT_TIMEOUT_SECONDS, type=int, help="MATLAB timeout in seconds.")
    args = parser.parse_args()

    summary = annotate_ecg_file(
        args.input,
        args.fs,
        output_dir=args.output_dir,
        matlab_bin=args.matlab_bin,
        timeout=args.timeout,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
