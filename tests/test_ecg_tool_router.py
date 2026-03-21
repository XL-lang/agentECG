import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
import sys

from mAgents.ecg_tool_router import compose_routing_context, route_ecg_analysis


_MODELS_MODULE = ModuleType("models")
_PYTORCH_MODULE = ModuleType("models.pytorch_inception")
_CLASSIFIER_MODULE = ModuleType("models.pytorch_inception.ecgqa_classifier")
_FIG_TOOL_MODULE = ModuleType("mAgents.fig_anaysis_tool")
_ECGDELI_TOOL_MODULE = ModuleType("mAgents.ecgdeli_analysis_tool")


class _ECGQAClassifierManager:
    pass


_CLASSIFIER_MODULE.ECGQAClassifierManager = _ECGQAClassifierManager


class _ECGFigAnaysiser:
    name = "ECG_fig_anaysis_tool"
    description = "stub visual ECG tool"


class _ECGDeliPrepareTool:
    name = "ECGDeli_prepare_tool"
    description = "stub ECGdeli prepare tool"


class _ECGDeliMeasurementTool:
    name = "ECGDeli_measurement_tool"
    description = "stub ECGdeli measurement tool"


_FIG_TOOL_MODULE.ECGFigAnaysiser = _ECGFigAnaysiser
_ECGDELI_TOOL_MODULE.ECGDeliPrepareTool = _ECGDeliPrepareTool
_ECGDELI_TOOL_MODULE.ECGDeliMeasurementTool = _ECGDeliMeasurementTool
sys.modules.setdefault("models", _MODELS_MODULE)
sys.modules.setdefault("models.pytorch_inception", _PYTORCH_MODULE)
sys.modules.setdefault("models.pytorch_inception.ecgqa_classifier", _CLASSIFIER_MODULE)
sys.modules.setdefault("mAgents.fig_anaysis_tool", _FIG_TOOL_MODULE)
sys.modules.setdefault("mAgents.ecgdeli_analysis_tool", _ECGDELI_TOOL_MODULE)


_PROMPT_SPEC = spec_from_file_location(
    "prompt_module",
    Path(__file__).resolve().parents[1] / "utils" / "prompt.py",
)
assert _PROMPT_SPEC is not None and _PROMPT_SPEC.loader is not None
_PROMPT_MODULE = module_from_spec(_PROMPT_SPEC)
_PROMPT_SPEC.loader.exec_module(_PROMPT_MODULE)
get_ecg_analysis_prompt = _PROMPT_MODULE.get_ecg_analysis_prompt


class DummyECG:
    def __init__(self, fs=None):
        self.fs = fs


class ECGToolRouterTests(unittest.TestCase):
    def test_routes_interval_question_to_ecgdeli(self):
        decision = route_ecg_analysis(
            question="What is the QRS duration and QT interval?",
            choices=["normal", "abnormal"],
            pre_analysis_res=None,
            ecgs={"ecg": DummyECG(fs=500)},
        )
        self.assertEqual(decision["preferred_tool"], "ECGDeli_measurement_tool")
        self.assertTrue(decision["requires_fs"])
        self.assertTrue(decision["fs_available"])

    def test_routes_morphology_question_to_visual_tool(self):
        decision = route_ecg_analysis(
            question="Does the tracing show morphology change and visual ST shape abnormality?",
            choices=["yes", "no"],
            pre_analysis_res=None,
            ecgs={"ecg": DummyECG(fs=500)},
        )
        self.assertEqual(decision["preferred_tool"], "ECG_fig_anaysis_tool")
        self.assertIn("morphology", decision["task_tags"])

    def test_routes_noise_question_to_visual_tool(self):
        decision = route_ecg_analysis(
            question="What types of noises are displayed in lead I in this ECG waveform?",
            choices=["baseline drift", "burst noise", "electrodes problems", "none", "static noise"],
            pre_analysis_res=None,
            ecgs={"ecg": DummyECG(fs=500)},
        )
        self.assertEqual(decision["preferred_tool"], "ECG_fig_anaysis_tool")
        self.assertIn("noise_sensitive", decision["task_tags"])
        self.assertIn("noise, artifact, or interference classification", decision["routing_constraints"])

    def test_marks_missing_fs_for_duration_tasks(self):
        decision = route_ecg_analysis(
            question="Measure the PR interval duration.",
            choices=["short", "normal", "long"],
            pre_analysis_res=None,
            ecgs={"ecg": DummyECG(fs=None)},
        )
        self.assertEqual(decision["preferred_tool"], "ECGDeli_measurement_tool")
        self.assertTrue(decision["requires_fs"])
        self.assertFalse(decision["fs_available"])
        self.assertIn("hard-coded sampling frequency", decision["routing_constraints"])

    def test_prompt_includes_routing_constraints(self):
        decision = route_ecg_analysis(
            question="Measure the RR interval.",
            choices=["regular", "irregular"],
            pre_analysis_res="Need interval evidence.",
            ecgs={"ecg": DummyECG(fs=None)},
        )
        prompt = get_ecg_analysis_prompt(
            "Measure the RR interval.",
            ["regular", "irregular"],
            "Need interval evidence.",
            ["ecg"],
            None,
            "Relevant micro-skills:\n- None retrieved from prior tasks.",
            compose_routing_context(decision),
        )
        self.assertIn("ECG tool routing verdict", prompt)
        self.assertIn("preferred_tool: ECGDeli_measurement_tool", prompt)
        self.assertIn("do not produce duration-based conclusions from hard-coded assumptions", prompt)
        self.assertIn("Extract that field once and store the raw string", prompt)


if __name__ == "__main__":
    unittest.main()
