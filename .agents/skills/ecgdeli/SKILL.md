---
name: ecgdeli
description: Run ECGdeli MATLAB-based ECG delineation and feature extraction from Codex. Use when an agent needs to annotate ECG wave fiducial points, extract ECGdeli amplitude or interval features, validate ECGdeli on local MATLAB, or call the bundled Python wrapper around MATLAB for ECG signal matrices or CSV files.
---

# ECGdeli

## Quick Start

Use `scripts/ecgdeli_wrapper.py` instead of calling MATLAB directly. The wrapper validates inputs, starts MATLAB with `-batch`, runs ECGdeli, and returns the parsed `summary.json`.

```python
from pathlib import Path
import sys

skill_dir = Path(".agents/skills/ecgdeli").resolve()
sys.path.insert(0, str(skill_dir / "scripts"))

from ecgdeli_wrapper import annotate_ecg_file

summary = annotate_ecg_file("input.csv", fs=500, output_dir="ecgdeli_out")
```

Input ECG matrices must be `samples x leads`. The CLI form is:

```bash
python .agents/skills/ecgdeli/scripts/ecgdeli_wrapper.py \
  --input input.csv \
  --fs 500 \
  --output-dir ecgdeli_out
```

## Workflow

1. Confirm MATLAB is licensed before analysis:
   `matlab -batch "disp(version)"`.
2. If MATLAB prompts for sign-in under WSL, run:
   `bash external/ECGdeli/activate_matlab_wslg.sh`.
3. Validate ECGdeli itself with:
   `bash external/ECGdeli/run_smoke_test_matlab.sh`.
4. For actual analysis, call `annotate_ecg(...)` for an in-memory matrix or `annotate_ecg_file(...)` for a CSV file.

## Outputs

The wrapper returns the JSON summary and writes full results into the output directory:

- `summary.json`: status, sample count, lead count, beat count, and output paths.
- `results.mat`: full fiducial points and extracted ECGdeli features.

Read `references/io.md` when exact input/output contracts matter.

## Failure Handling

- If MATLAB cannot be found, pass `matlab_bin="/home/xl/MATLAB/R2025b/bin/matlab"` or fix PATH.
- If MATLAB asks for a MathWorks account, activation is incomplete; do not collect credentials in chat.
- If ECGdeli is missing, restore `/home/xl/agentECG/external/ECGdeli` or set `ECGDELI_ROOT`.
- If MATLAB fails, inspect the raised error message; it includes return code plus stdout/stderr tails.
