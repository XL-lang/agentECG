import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
import sys


_MODELS_MODULE = ModuleType("models")
_PYTORCH_MODULE = ModuleType("models.pytorch_inception")
_CLASSIFIER_MODULE = ModuleType("models.pytorch_inception.ecgqa_classifier")
_FIG_TOOL_MODULE = ModuleType("mAgents.fig_anaysis_tool")
_ECGDELI_TOOL_MODULE = ModuleType("mAgents.ecgdeli_analysis_tool")


class _ECGQAClassifierManager:
    pass


class _ECGFigAnaysiser:
    name = "ECG_fig_anaysis_tool"
    description = "stub visual ECG tool"


class _ECGDeliPrepareTool:
    name = "ECGDeli_prepare_tool"
    description = "stub ECGdeli prepare tool"


class _ECGDeliMeasurementTool:
    name = "ECGDeli_measurement_tool"
    description = "stub ECGdeli measurement tool"


_CLASSIFIER_MODULE.ECGQAClassifierManager = _ECGQAClassifierManager
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
get_pathology_inquiry_prompt = _PROMPT_MODULE.get_pathology_inquiry_prompt


class PreAnalysisPromptTests(unittest.TestCase):
    def test_prompt_enforces_local_first_before_remote_search(self):
        prompt = get_pathology_inquiry_prompt(
            "Which leads show non-specific st changes?",
            ["lead I", "lead II", "none"],
            None,
            skill_context="Local reusable skill plan:\n- analyze-st-segment-morphology",
            local_pre_analysis_context="relevant_skills:\n- analyze-st-segment-morphology\nfinal_search_needed: yes",
            remote_search_needed=True,
            open_questions_for_remote_search=["Confirm lead-specific interpretation rules."],
        )
        self.assertIn("Local-first pre-analysis must be performed before any remote search.", prompt)
        self.assertIn("Only use remote search as a supplement", prompt)
        self.assertIn("do not search for or include Python code snippets", prompt)
        self.assertIn("Do not call `ECG_fig_anaysis_tool`, `ECGDeli_prepare_tool`, or `ECGDeli_measurement_tool`", prompt)

    def test_prompt_can_disable_remote_search(self):
        prompt = get_pathology_inquiry_prompt(
            "Measure the QRS duration.",
            ["normal", "wide"],
            None,
            skill_context="Local reusable skill plan:\n- analyze-qrs-duration",
            local_pre_analysis_context="relevant_skills:\n- analyze-qrs-duration\nfinal_search_needed: no",
            remote_search_needed=False,
            open_questions_for_remote_search=[],
        )
        self.assertIn("Do not call remote search tools.", prompt)
        self.assertIn("final_search_needed: no", prompt)


if __name__ == "__main__":
    unittest.main()
