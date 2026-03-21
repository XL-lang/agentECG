---
name: ecgdeli-analysis
description: Use fine-grained ECGdeli measurement primitives from a cached session. Prefer this skill when an agent needs to cut one wave, retrieve one fiducial point, measure one interval, or fetch one amplitude or height from a samples x leads ECG matrix.
---

# ECGdeli Analysis

## Trigger

Use this skill when the task depends on beat-level structured ECG delineation, especially for:

- slicing a single `p_wave`, `qrs_complex`, `t_wave`, `pr_interval`, `qt_interval`, or `st_segment`
- retrieving one fiducial point such as `p_onset`, `r_peak`, `qrs_offset`, or `t_offset`
- measuring a single interval such as `qrs_duration`, `qt_interval`, `rr_interval`, or `qtc_interval`
- measuring one amplitude or height such as `p_amplitude`, `r_amplitude`, or `t_wave_height`
- building evidence from one beat and one lead at a time

Prefer `ECG_fig_anaysis_tool` instead when the task is mainly visual, noise-sensitive, or depends on qualitative morphology that is fragile to rule-based measurement.
Do not treat the visual tool as the primary segment slicer when ECGdeli can provide the needed wave bounds or slices.

## Standard Workflow

1. Validate the input shape is `samples x leads` and confirm `fs > 0`.
2. Call `prepare_ecgdeli_session(...)` or `prepare_ecgdeli_file_session(...)` once.
   - `ECGDeli_prepare_tool` returns a session summary payload containing a `session_id` field.
   - Extract the `session_id` value once and pass only that raw string to `ECGDeli_measurement_tool`.
   - Do not pass the entire summary payload as `session_id`, and do not index a string `session_id` again.
3. Reuse the returned session with one or more narrow queries:
   - `get_fiducial_point(...)`
   - `get_wave_bounds(...)`
   - `slice_wave(...)`
   - `get_interval_value(...)`
   - `get_amplitude_value(...)`
   - `get_height(...)`
4. Only combine multiple fine-grained measurements when the task truly needs a higher-level summary.
5. `slice_wave(...)` returns `samples_summary` by default. Request full `samples` only when downstream logic truly needs the raw waveform values.

## Python API

```python
from pathlib import Path
import sys

skill_dir = Path(".agents/skills/ecgdeli-analysis").resolve()
sys.path.insert(0, str(skill_dir / "scripts"))

from ecgdeli_analysis import (
    prepare_ecgdeli_session,
    prepare_ecgdeli_file_session,
    get_fiducial_point,
    get_wave_bounds,
    slice_wave,
    get_interval_value,
    get_amplitude_value,
    get_height,
)
```

Example:

```python
session = prepare_ecgdeli_session(signal, fs=500)
session_id = session.session_id
qrs_bounds = get_wave_bounds(session, beat_index=0, wave_name="qrs_complex", lead_index=0)
qrs_duration = get_interval_value(session, beat_index=0, feature_name="qrs_duration", lead_index=0)
r_amp = get_amplitude_value(session, beat_index=0, feature_name="r_amplitude", lead_index=0)
```

## Failure Handling

- If MATLAB is missing, pass `matlab_bin="/home/xl/MATLAB/R2025b/bin/matlab"` or fix `PATH`.
- If ECGdeli is not under `external/ECGdeli`, set `ECGDELI_ROOT`.
- If MATLAB execution fails, the preparation call raises a structured error payload.
- Fine-grained measurement calls return `status: "error"` with `error_identifier` and `error_message` instead of silently guessing.
- Read `references/io.md` only when you need the exact session and result contract.
