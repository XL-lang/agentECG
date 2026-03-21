# ECGdeli Analysis I/O Contract

## Inputs

- `signal`: numeric matrix shaped `samples x leads`
- `input_csv`: CSV file containing the same `samples x leads` matrix
- `fs`: positive sampling rate in Hz
- `lead_index`: `0-based int`
- `beat_index`: `0-based int`
- `wave_name`: one of `p_wave`, `qrs_complex`, `t_wave`, `pr_interval`, `pq_interval`, `qt_interval`, `st_segment`
- `feature_name`: one of:
  - leadwise timing: `p_duration`, `qrs_duration`, `t_duration`, `pq_interval`, `pr_interval`, `qt_interval`, `rr_interval`
  - synchronized timing: previous list plus `qtc_interval`
  - amplitude: `p_amplitude`, `q_amplitude`, `r_amplitude`, `s_amplitude`, `t_amplitude`

The wrapper does not infer sampling rate automatically.

## Session Preparation

`prepare_ecgdeli_session(...)` and `prepare_ecgdeli_file_session(...)` return an `ECGDeliSession` with:

- cached raw signal
- cached filtered signal from ECGdeli preprocessing
- `fptMultiChannel` and per-lead `fptCell`
- timing and amplitude feature arrays
- artifact paths such as `results.mat` and `summary.json`

## Fine-Grained Results

### `measurement_result`

- `status`
- `value`
- `units`
- `lead_index`
- `beat_index`
- `feature_name`
- `source`
- `valid`
- `error_identifier` / `error_message` on failure

### `bounds_result`

- `status`
- `start_index`
- `end_index`
- `length_samples`
- `length_ms`
- `lead_index`
- `beat_index`
- `wave_name`
- `valid`
- `error_identifier` / `error_message` on failure

### `wave_slice_result`

- `status`
- `samples_summary`
- `start_index`
- `end_index`
- `length_samples`
- `length_ms`
- `lead_index`
- `beat_index`
- `wave_name`
- `valid`
- `samples` only when `include_samples=True`
- `error_identifier` / `error_message` on failure

## Fiducial Point Mapping

`fptMultiChannel` and each lead table in `fptCell` use these 13 columns:

1. `p_onset`
2. `p_peak`
3. `p_offset`
4. `qrs_onset`
5. `q_peak`
6. `r_peak`
7. `s_peak`
8. `qrs_offset`
9. `j_point`
10. `t_onset`
11. `t_peak`
12. `t_offset`
13. `class_id`

## Error Behavior

- Python-side validation raises `ValueError` for invalid ECG matrices or sampling rates during session preparation.
- MATLAB-side preparation failures raise `RuntimeError` with a structured JSON payload.
- Fine-grained query functions return `status: "error"` payloads when indices, wave names, feature names, or prerequisites are invalid.
