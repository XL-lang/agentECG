import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

_SPEC = spec_from_file_location(
    "ecgdeli_analysis_module",
    Path(__file__).resolve().parents[1] / ".agents" / "skills" / "ecgdeli-analysis" / "scripts" / "ecgdeli_analysis.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_MEMORY_SPEC = spec_from_file_location(
    "memory_manage_module",
    Path(__file__).resolve().parents[1] / "utils" / "memory_manage.py",
)
assert _MEMORY_SPEC is not None and _MEMORY_SPEC.loader is not None
_MEMORY_MODULE = module_from_spec(_MEMORY_SPEC)
sys.modules[_MEMORY_SPEC.name] = _MEMORY_MODULE
_MEMORY_SPEC.loader.exec_module(_MEMORY_MODULE)

ECGDeliSession = _MODULE.ECGDeliSession
get_amplitude_value = _MODULE.get_amplitude_value
get_height = _MODULE.get_height
get_interval_value = _MODULE.get_interval_value
get_wave_bounds = _MODULE.get_wave_bounds
slice_wave = _MODULE.slice_wave
ActionStep = _MEMORY_MODULE.ActionStep
clean_memory = _MEMORY_MODULE.clean_memory
summarize_array_like = _MEMORY_MODULE.summarize_array_like


class ECGDeliGranularTests(unittest.TestCase):
    def setUp(self):
        signal = np.arange(60, dtype=float).reshape(30, 2)
        filtered = signal + 0.5
        fpt_by_lead = [
            np.array(
                [
                    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0],
                    [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 0],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0],
                    [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 0],
                ],
                dtype=int,
            ),
        ]
        self.session = ECGDeliSession(
            session_id="session-1",
            signal=signal,
            filtered_signal=filtered,
            fs=1000.0,
            work_dir="/tmp/ecgdeli",
            results_mat="/tmp/ecgdeli/results.mat",
            summary_json="/tmp/ecgdeli/summary.json",
            fpt_multichannel=fpt_by_lead[0].copy(),
            fpt_by_lead=fpt_by_lead,
            timing_features=np.array(
                [
                    [[100, 80, 120, 130, 150, 200, 900], [110, 85, 125, 135, 155, 205, 910]],
                    [[101, 81, 121, 131, 151, 201, 901], [111, 86, 126, 136, 156, 206, 911]],
                ],
                dtype=float,
            ),
            timing_features_sync=np.array(
                [[102, 82, 122, 132, 152, 202, 210, 902], [112, 87, 127, 137, 157, 207, 215, 912]],
                dtype=float,
            ),
            amplitude_features=np.array(
                [
                    [[0.1, -0.2, 1.0, -0.3, 0.4], [0.11, -0.21, 1.01, -0.31, 0.41]],
                    [[0.2, -0.1, 0.9, -0.2, 0.5], [0.21, -0.11, 0.91, -0.21, 0.51]],
                ],
                dtype=float,
            ),
            summary={},
        )

    def test_wave_bounds_and_slice(self):
        bounds = get_wave_bounds(self.session, beat_index=0, wave_name="qrs_complex", lead_index=0)
        self.assertEqual(bounds["status"], "ok")
        self.assertEqual(bounds["start_index"], 5)
        self.assertEqual(bounds["end_index"], 9)
        self.assertEqual(bounds["length_samples"], 4)

        sliced = slice_wave(self.session, beat_index=0, wave_name="qrs_complex", lead_index=0)
        self.assertEqual(sliced["status"], "ok")
        self.assertNotIn("samples", sliced)
        self.assertEqual(sliced["samples_summary"]["__type__"], "array_summary")
        self.assertEqual(sliced["samples_summary"]["shape"], [4])
        self.assertEqual(sliced["samples_summary"]["preview"], [10.5, 12.5, 14.5, 16.5])

        sliced_full = slice_wave(
            self.session,
            beat_index=0,
            wave_name="qrs_complex",
            lead_index=0,
            include_samples=True,
        )
        self.assertEqual(len(sliced_full["samples"]), 4)
        self.assertEqual(sliced_full["samples"][0], self.session.filtered_signal[5, 0])

    def test_interval_measurements(self):
        leadwise = get_interval_value(self.session, beat_index=0, feature_name="qrs_duration", lead_index=1)
        self.assertEqual(leadwise["status"], "ok")
        self.assertEqual(leadwise["value"], 81.0)
        self.assertEqual(leadwise["units"], "ms")

        sync = get_interval_value(
            self.session, beat_index=1, feature_name="qtc_interval", synchronized=True
        )
        self.assertEqual(sync["status"], "ok")
        self.assertEqual(sync["value"], 215.0)

    def test_amplitude_and_height(self):
        amp = get_amplitude_value(self.session, beat_index=0, feature_name="r_amplitude", lead_index=0)
        self.assertEqual(amp["status"], "ok")
        self.assertEqual(amp["value"], 1.0)

        height = get_height(self.session, beat_index=1, wave_name="t_wave", lead_index=1)
        self.assertEqual(height["status"], "ok")
        self.assertEqual(height["value"], 0.51)
        self.assertEqual(height["feature_name"], "t_wave_height")

    def test_invalid_indices_return_error_payload(self):
        result = get_wave_bounds(self.session, beat_index=99, wave_name="p_wave", lead_index=0)
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["valid"])

    def test_long_slice_returns_summary_and_can_include_full_samples(self):
        session = ECGDeliSession(
            session_id="session-long",
            signal=np.arange(500, dtype=float).reshape(250, 2),
            filtered_signal=np.arange(500, dtype=float).reshape(250, 2) + 1.0,
            fs=500.0,
            work_dir="/tmp/ecgdeli",
            results_mat="/tmp/ecgdeli/results.mat",
            summary_json="/tmp/ecgdeli/summary.json",
            fpt_multichannel=np.array([[10, 12, 18, 20, 21, 25, 27, 60, 60, 65, 70, 75, 0]], dtype=int),
            fpt_by_lead=[np.array([[10, 12, 18, 20, 21, 25, 27, 60, 60, 65, 70, 75, 0]], dtype=int)] * 2,
            timing_features=np.zeros((2, 1, 7), dtype=float),
            timing_features_sync=np.zeros((1, 8), dtype=float),
            amplitude_features=np.zeros((2, 1, 5), dtype=float),
            summary={},
        )
        sliced = slice_wave(session, beat_index=0, wave_name="qrs_complex", lead_index=0)
        self.assertEqual(sliced["status"], "ok")
        self.assertEqual(sliced["samples_summary"]["__type__"], "array_summary")
        self.assertEqual(sliced["samples_summary"]["shape"], [40])
        self.assertNotIn("samples", sliced)

        sliced_full = slice_wave(session, beat_index=0, wave_name="qrs_complex", lead_index=0, include_samples=True)
        self.assertEqual(len(sliced_full["samples"]), 40)
        self.assertEqual(sliced_full["samples_summary"]["__type__"], "array_summary")

    def test_memory_cleaner_summarizes_long_observation_arrays(self):
        long_payload = {"samples": list(range(40)), "short": [1, 2, 3]}
        step = ActionStep()
        step.error = None
        step.observations = long_payload
        agent = SimpleNamespace(memory=SimpleNamespace(steps=[step]))

        clean_memory(step, agent)

        self.assertEqual(step.observations["samples"]["__type__"], "array_summary")
        self.assertEqual(step.observations["samples"]["shape"], [40])
        self.assertEqual(step.observations["short"], [1, 2, 3])

    def test_summarize_array_like_non_numeric_sequence(self):
        summary = summarize_array_like(["a"] * 40)
        self.assertEqual(summary["__type__"], "array_summary")
        self.assertEqual(summary["shape"], [40])
        self.assertIsNone(summary["stats"])


if __name__ == "__main__":
    unittest.main()
