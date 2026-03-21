# ECGdeli Wrapper I/O

## Inputs

Use a numeric ECG matrix with shape `samples x leads`.

- `annotate_ecg(signal, fs, output_dir=None, matlab_bin="matlab")`
  accepts a Python sequence or NumPy-like array. The wrapper writes it to CSV before calling MATLAB.
- `annotate_ecg_file(input_csv, fs, output_dir=None, matlab_bin="matlab")`
  accepts a CSV file readable by MATLAB `readmatrix`.
- `fs` is the sampling rate in Hz and must be positive.

The local ECGdeli checkout is resolved from `ECGDELI_ROOT` when set. Otherwise it defaults to:

```text
/home/xl/agentECG/external/ECGdeli
```

## Outputs

The output directory contains:

- `summary.json`: JSON summary with `status`, `samples`, `leads`, `beats`, `results_mat`, and `summary_json`.
- `results.mat`: MATLAB data containing:
  - `fptMultiChannel`
  - `fptCell`
  - `amplitudeFeatures`
  - `timingFeatures`
  - `timingFeaturesSync`
  - `fs`
  - `inputCsv`

`annotate_ecg(...)` and `annotate_ecg_file(...)` return the parsed contents of `summary.json`.

## Error Behavior

Python-side validation raises `ValueError` for invalid ECG matrices or sampling rates.

Execution failures raise `RuntimeError` with:

- MATLAB command path.
- MATLAB return code.
- Tail of stdout and stderr.

MATLAB also writes `summary.json` with `status: "error"` when it can create the output directory before failing.
