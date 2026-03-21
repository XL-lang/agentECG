import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

_PRE_ANALYSIS_MODULE = ModuleType("mAgents.pre_analysis_agent")
_DATA_ANALYSIS_MODULE = ModuleType("mAgents.data_analysis_agent")
_MODEL_AGENT_MODULE = ModuleType("mAgents.model_agent")
_ROUTER_MODULE = ModuleType("mAgents.ecg_tool_router")
_CHECKER_MODULE = ModuleType("mAgents.llm_checker")
_SLICE_AUDIT_MODULE = ModuleType("agent_reflect.vlm.slice_audit")
_UTILS_MODULE = ModuleType("utils")
_LOGGER_MODULE = ModuleType("utils.logger")
_PROMPT_MODULE = ModuleType("utils.prompt")


def _noop_agent_factory():
    return None


class _NoopChecker:
    def __init__(self, choices):
        self.choices = choices

    def check(self, *args, **kwargs):
        return True


class _NoopAuditor:
    def __init__(self, fs=None):
        self.fs = fs

    def audit(self, **kwargs):
        return {"verdict": "uncertain", "reason": "stub"}


_PRE_ANALYSIS_MODULE.create_pre_analysis_agent = _noop_agent_factory
_DATA_ANALYSIS_MODULE.create_data_analysis_agent = _noop_agent_factory
_MODEL_AGENT_MODULE.create_model_agent = _noop_agent_factory
_ROUTER_MODULE.compose_routing_context = lambda decision: str(decision)
_ROUTER_MODULE.route_ecg_analysis = lambda **kwargs: {"preferred_tool": "none", "reason": ""}
_CHECKER_MODULE.ECGQA_EMA_checker = _NoopChecker
_SLICE_AUDIT_MODULE.SegmentSliceAuditor = _NoopAuditor
_SLICE_AUDIT_MODULE.capture_agent_variables = lambda *args, **kwargs: {"summaries": []}
_LOGGER_MODULE.get_logger = lambda: type(
    "Logger",
    (),
    {"warning": lambda *args, **kwargs: None, "error": lambda *args, **kwargs: None},
)()
_PROMPT_MODULE.get_pathology_inquiry_prompt = lambda *args, **kwargs: "pre-analysis prompt"
_PROMPT_MODULE.get_ecg_analysis_prompt = lambda *args, **kwargs: "data-analysis prompt"
_PROMPT_MODULE.get_ecgSignals_doc_prompt = lambda: "ECG API prompt"
_PROMPT_MODULE.get_local_models = lambda: "Local tool prompt"
_UTILS_MODULE.logger = _LOGGER_MODULE
_UTILS_MODULE.prompt = _PROMPT_MODULE

sys.modules.setdefault("mAgents.pre_analysis_agent", _PRE_ANALYSIS_MODULE)
sys.modules.setdefault("mAgents.data_analysis_agent", _DATA_ANALYSIS_MODULE)
sys.modules.setdefault("mAgents.model_agent", _MODEL_AGENT_MODULE)
sys.modules.setdefault("mAgents.ecg_tool_router", _ROUTER_MODULE)
sys.modules.setdefault("mAgents.llm_checker", _CHECKER_MODULE)
sys.modules.setdefault("agent_reflect.vlm.slice_audit", _SLICE_AUDIT_MODULE)
sys.modules.setdefault("utils", _UTILS_MODULE)
sys.modules.setdefault("utils.logger", _LOGGER_MODULE)
sys.modules.setdefault("utils.prompt", _PROMPT_MODULE)

from thread_executor import AgentExecutionOptions, AgentTask
import agent_runner
from agent_reflect.skills.hierarchy import (
    PendingSkillGenerator,
    ReflectiveSkillGenerator,
    ReflectionGenerator,
    SkillEvaluator,
    SkillRegistry,
    SkillUnit,
)


def _make_skill(
    *,
    skill_id: str,
    name: str,
    procedure_summary: str,
    feature_scope: str,
    status: str = "active",
    applies_on_retry_only: bool = False,
) -> SkillUnit:
    return SkillUnit(
        skill_id=skill_id,
        agent_name="data_analysis_agent",
        name=name,
        description=f"skill for {feature_scope}",
        level="micro",
        category="reflective-analysis",
        input_requirements=["ecg object"],
        procedure_summary=procedure_summary,
        evidence_pattern=[feature_scope],
        tool_dependencies=[],
        applicability=f"use for {feature_scope}",
        failure_modes=["wrong answer mapping"],
        source_sample_ids=["s1"],
        confidence=0.8,
        reuse_count=0,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        feature_scope=feature_scope,
        action_bundle=["read waveform", "map evidence to answer"],
        validation_basis=["final_answer_match"],
        status=status,
        applies_on_retry_only=applies_on_retry_only,
    )


class ReflectiveSkillTests(unittest.TestCase):
    def test_load_legacy_registry_populates_new_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "agent_skill_registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "skills": [
                            {
                                "skill_id": "legacy1",
                                "agent_name": "data_analysis_agent",
                                "name": "legacy",
                                "description": "legacy description",
                                "level": "micro",
                                "category": "measurement",
                                "input_requirements": ["ecg object"],
                                "procedure_summary": "Call a documented ECG method and validate one feature.",
                                "evidence_pattern": ["segment"],
                                "tool_dependencies": [],
                                "applicability": "legacy applicability",
                                "failure_modes": ["noise"],
                                "source_sample_ids": ["s1"],
                                "confidence": 0.7,
                                "reuse_count": 0,
                                "created_at": "2026-01-01T00:00:00+00:00",
                                "updated_at": "2026-01-01T00:00:00+00:00",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            registry = SkillRegistry(registry_path).load()
            skill = registry.skills[0]
            self.assertEqual(skill.skill_source, "success_trace")
            self.assertEqual(skill.feature_scope, "")
            self.assertEqual(skill.action_bundle, [])
            self.assertEqual(skill.validation_basis, [])
            self.assertEqual(skill.status, "active")
            self.assertEqual(skill.origin_prediction, "")
            self.assertEqual(skill.origin_gold, [])
            self.assertFalse(skill.applies_on_retry_only)

    def test_pending_review_skills_are_not_retrieved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = SkillRegistry(Path(tmpdir) / "agent_skill_registry.json").load()
            active = _make_skill(
                skill_id="active1",
                name="analyze-qrs-duration",
                procedure_summary="Feature workflow: verify `qrs_duration` with segment evidence and map the result to the allowed answer.",
                feature_scope="qrs_duration",
            )
            pending = _make_skill(
                skill_id="pending1",
                name="retry-correct-qrs-duration",
                procedure_summary="Retry-only correction workflow: verify `qrs_duration` and re-check the failed answer path.",
                feature_scope="qrs_duration",
                status="pending_review",
            )
            registry.skills = [active, pending]
            results = registry.find_relevant_skills("Measure qrs_duration", top_k=5)
            self.assertEqual([skill.skill_id for skill in results], ["active1"])

    def test_conflict_writes_pending_review_json_and_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = SkillRegistry(Path(tmpdir) / "agent_skill_registry.json").load()
            existing = _make_skill(
                skill_id="skill_a",
                name="analyze-qrs-duration",
                procedure_summary="Feature workflow: verify `qrs_duration` with verified fs and map the result to the allowed answer.",
                feature_scope="qrs_duration",
            )
            candidate = _make_skill(
                skill_id="skill_b",
                name="analyze-qrs-duration-alt",
                procedure_summary="Feature workflow: verify `qrs_duration` with a conflicting threshold and map the result to the allowed answer.",
                feature_scope="qrs_duration",
            )
            registry.skills = [existing]
            registry.semantic_merger.find_semantic_relation = lambda skill, existing_skills: (
                existing,
                {"decision": "conflict", "reason": "threshold direction mismatch"},
            )

            stored = registry.upsert_skill(candidate)
            registry.save()

            self.assertEqual(stored.status, "pending_review")
            self.assertTrue(registry.conflict_table_path.exists())
            self.assertTrue(registry.pending_review_path.exists())
            pending_reviews = json.loads(registry.pending_review_path.read_text(encoding="utf-8"))
            self.assertEqual(pending_reviews[0]["feature_scope"], "qrs_duration")
            self.assertTrue(pending_reviews[0]["from_failure_retry"] is False)

    def test_reflective_generator_builds_success_and_retry_skills(self):
        generator = ReflectiveSkillGenerator()
        memory = [
            {
                "tool": "ECG_fig_anaysis_tool",
                "code": "ecg.get_lead_signals('II'); ECG_fig_anaysis_tool(data=lead_ii, question='Does this ECG show t-wave inversion?')",
            }
        ]
        success_skills = generator.generate(
            sample_id="s1",
            question="Does this ECG show t-wave inversion?",
            choices=["yes", "no"],
            data_analysis_memory=memory,
            final_answer="yes",
            mode="success",
        )
        retry_skills = generator.generate(
            sample_id="s1",
            question="Does this ECG show t-wave inversion?",
            choices=["yes", "no"],
            data_analysis_memory=memory,
            final_answer=None,
            predicted_answer="no",
            gold_answers=["yes"],
            mode="failure_retry",
        )
        self.assertEqual(success_skills[0].status, "active")
        self.assertEqual(success_skills[0].skill_source, "success_trace")
        self.assertFalse(success_skills[0].applies_on_retry_only)
        self.assertGreaterEqual(len(success_skills[0].action_bundle), 2)
        self.assertEqual(retry_skills[0].status, "retry_only")
        self.assertEqual(retry_skills[0].skill_source, "failure_retry")
        self.assertTrue(retry_skills[0].applies_on_retry_only)
        self.assertEqual(retry_skills[0].origin_gold, ["yes"])

    def test_reflective_generator_filters_native_capability_duplicates(self):
        generator = ReflectiveSkillGenerator()
        memory = [
            {
                "tool": "ECGDeli_measurement_tool",
                "code": "ecg.get_lead_signals('II'); ecg.get_lead_segment('II', 'QRS complex'); ECGDeli_measurement_tool(session_id='s', operation='get_interval_value', feature_name='qrs_duration'); fs = 500",
            }
        ]
        skills = generator.generate(
            sample_id="s1",
            question="What is the QRS duration?",
            choices=["normal", "wide"],
            data_analysis_memory=memory,
            final_answer="wide",
            mode="success",
        )
        self.assertEqual(skills, [])

    def test_evaluator_accepts_active_reflective_skill(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = SkillRegistry(Path(tmpdir) / "agent_skill_registry.json").load()
            skill = _make_skill(
                skill_id="skill_active",
                name="analyze-t-wave-inversion",
                procedure_summary="Feature workflow: verify `t_wave_inversion` from the narrowest T-wave evidence path and map the result to the allowed answer.",
                feature_scope="t_wave_inversion",
            )
            accepted, rejected = SkillEvaluator().evaluate([skill], registry)
            self.assertEqual(len(accepted), 1)
            self.assertEqual(rejected, [])

    def test_evaluator_rejects_native_capability_duplicate_reflective_skill(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = SkillRegistry(Path(tmpdir) / "agent_skill_registry.json").load()
            skill = _make_skill(
                skill_id="skill_dup",
                name="analyze-qrs-duration",
                procedure_summary="Feature workflow: verify `qrs_duration` with the smallest valid evidence path, then read target lead waveform before drawing a feature conclusion; slice the narrowest ECG segment associated with the feature; consume ECGdeli feature-level measurement output for the target feature; verify sampling frequency and convert sample span into duration only after confirmation.",
                feature_scope="qrs_duration",
            )
            skill.tool_dependencies = ["ECGDeli_measurement_tool"]
            skill.action_bundle = [
                "read target lead waveform before drawing a feature conclusion",
                "slice the narrowest ECG segment associated with the feature",
                "consume ECGdeli feature-level measurement output for the target feature",
                "verify sampling frequency and convert sample span into duration only after confirmation",
            ]
            accepted, rejected = SkillEvaluator().evaluate([skill], registry)
            self.assertEqual(accepted, [])
            self.assertEqual(rejected, [("analyze-qrs-duration", "duplicates native tool/static skill capability")])

    def test_reflection_uses_vlm_morphology_support_signal(self):
        reflection = ReflectionGenerator().generate(
            question="Does this show T-wave inversion?",
            predicted_answer="no",
            gold_answers=["yes"],
            is_correct=False,
            data_analysis_memory=[{"code": "ecg.get_lead_segment('V5', 'T wave')"}],
            used_skill_names=[],
            learned_skill_names=[],
            variable_summaries=[{"name": "t_slice"}],
            vlm_slice_audit={
                "verdict": "valid",
                "reason": "slice is usable",
                "morphology_assessment": "the T-wave remains upright and does not support inversion",
                "supports_target_feature": False,
            },
        )
        self.assertIn("does not support the target feature", reflection["root_cause"])
        self.assertIn("upright", reflection["root_cause"])

    def test_pending_skill_generator_mentions_morphology_assessment(self):
        retry_context = PendingSkillGenerator().generate(
            question="Does this show T-wave inversion?",
            choices=["yes", "no"],
            failed_answer="yes",
            failure_reflection={"root_cause": "mapped evidence incorrectly"},
            data_analysis_memory=[{"code": "ecg.get_lead_segment('V5', 'T wave')"}],
            variable_summaries=[{"name": "t_slice"}],
            vlm_slice_audit={
                "verdict": "valid",
                "reason": "slice is usable",
                "morphology_assessment": "the observed T-wave is upright rather than inverted",
                "supports_target_feature": False,
            },
        )
        self.assertIn("Morphology assessment", retry_context)
        self.assertIn("did not clearly support the target feature", retry_context)


class AgentRunnerModeTests(unittest.TestCase):
    def test_run_pre_analysis_agent_returns_local_plan_when_remote_not_needed(self):
        class DummyAgent:
            def run(self, prompt):
                raise AssertionError("remote pre-analysis should be skipped")

        result = agent_runner._run_pre_analysis_agent(
            question="Measure QRS duration.",
            choices=["normal", "wide"],
            model_agent_res=None,
            pre_analysis_agent=DummyAgent(),
            skill_context="Local reusable skill plan:\n- analyze-qrs-duration",
            local_pre_analysis_context={
                "plan_text": "target_features:\n- qrs_duration\nfinal_search_needed: no",
                "remote_search_needed": False,
                "open_questions_for_remote_search": [],
            },
        )
        self.assertIn("final_search_needed: no", result)

    def test_build_local_pre_analysis_context_collects_remote_questions(self):
        context = agent_runner._build_local_pre_analysis_context(
            question="Which leads show non-specific st changes and what diagnostic criteria apply?",
            choices=["lead I", "lead II", "none"],
            skill_context=None,
            routing_decision={"preferred_tool": "ECG_fig_anaysis_tool", "reason": "morphology-heavy", "fallback_tool": "none", "requires_fs": False, "fs_available": True, "task_tags": ["morphology"], "routing_constraints": "", "prerequisites": {"single_ecg": True, "supports_matrix_input": True}},
        )
        self.assertTrue(context["remote_search_needed"])
        self.assertTrue(any("diagnostic criteria" in item.lower() or "definitions" in item.lower() for item in context["open_questions_for_remote_search"]))

    def test_eval_mode_skips_skill_retrieval_and_retry(self):
        class DummyAgent:
            def __init__(self):
                self.state = {}
                self.memory = self

            def get_full_steps(self):
                return []

        task = AgentTask(
            iteration=1,
            sample={
                "sample_id": "s1",
                "question": "Is this normal?",
                "template_answer": ["yes", "no"],
                "answer": ["yes"],
                "ecg_datas": [object()],
            },
            options=AgentExecutionOptions(run_mode="eval", save_agent_mem=False),
        )

        with patch.object(agent_runner, "create_pre_analysis_agent", return_value=DummyAgent()), \
            patch.object(agent_runner, "create_data_analysis_agent", return_value=DummyAgent()), \
            patch.object(agent_runner, "create_model_agent", return_value=DummyAgent()), \
            patch.object(agent_runner, "_retrieve_skill_context", side_effect=AssertionError("should not retrieve skills in eval mode")), \
            patch.object(agent_runner, "_run_model_agent", return_value=None), \
            patch.object(agent_runner, "_run_pre_analysis_agent", return_value="pre"), \
            patch.object(agent_runner, "_run_data_analysis_agent", return_value="yes"), \
            patch.object(agent_runner, "route_ecg_analysis", return_value={"preferred_tool": "none", "reason": ""}):
            result = agent_runner.process_sample(task)

        self.assertIsNone(result.error)
        self.assertEqual(result.result_entry["run_mode"], "eval")
        self.assertFalse(result.result_entry["retry_used"])
        self.assertEqual(result.result_entry["learned_data_analysis_skills"], [])
        self.assertEqual(result.result_entry["data_analysis_skill_retrieval_mode"], "disabled")

    def test_skill_build_segmentation_error_skips_reflection_and_retry(self):
        class DummyAgent:
            def __init__(self):
                self.state = {}
                self.memory = self

            def get_full_steps(self):
                return []

        task = AgentTask(
            iteration=1,
            sample={
                "sample_id": "s1",
                "question": "Measure the QRS duration.",
                "template_answer": ["normal", "wide"],
                "answer": ["wide"],
                "ecg_datas": [object()],
            },
            options=AgentExecutionOptions(run_mode="skill_build", save_agent_mem=False),
        )

        with patch.object(agent_runner, "create_pre_analysis_agent", return_value=DummyAgent()), \
            patch.object(agent_runner, "create_data_analysis_agent", return_value=DummyAgent()), \
            patch.object(agent_runner, "create_model_agent", return_value=DummyAgent()), \
            patch.object(agent_runner, "_retrieve_skill_context", return_value=("Relevant micro-skills:\n- None retrieved from prior tasks.", [], "disabled")), \
            patch.object(agent_runner, "_run_model_agent", return_value=None), \
            patch.object(agent_runner, "_run_pre_analysis_agent", return_value="pre"), \
            patch.object(agent_runner, "_run_data_analysis_agent", return_value="normal") as run_data_analysis, \
            patch.object(agent_runner, "route_ecg_analysis", return_value={"preferred_tool": "none", "reason": ""}), \
            patch.object(agent_runner, "capture_agent_variables", return_value={"summaries": [{"name": "qrs_slice"}]}), \
            patch.object(agent_runner, "_get_agent_memory", return_value=[{"code": "ecg.get_lead_segment('II', 'QRS complex')"}]), \
            patch.object(agent_runner, "_generate_reflective_skills", side_effect=AssertionError("should not generate retry skills after segmentation_error")), \
            patch.object(agent_runner, "_store_reflective_skills", side_effect=AssertionError("should not store skills after segmentation_error")), \
            patch.object(agent_runner, "ReflectionGenerator", side_effect=AssertionError("should not reflect after segmentation_error")), \
            patch.object(agent_runner, "SegmentSliceAuditor") as auditor_cls:
            auditor_cls.return_value.audit.return_value = {
                "verdict": "segmentation_error",
                "reason": "window misses qrs onset",
                "checked_variable": "qrs_slice",
                "segment_type": "qrs",
                "confidence": 0.95,
                "morphology_assessment": "the slice starts after q onset and cannot support qrs duration",
                "supports_target_feature": False,
                "checked": [],
            }
            result = agent_runner.process_sample(task)

        self.assertIsNone(result.error)
        self.assertTrue(result.result_entry["reflection_skipped"])
        self.assertEqual(result.result_entry["reflection_skip_reason"], "segmentation_error")
        self.assertFalse(result.result_entry["retry_used"])
        self.assertEqual(result.result_entry["learned_data_analysis_skills"], [])
        self.assertEqual(run_data_analysis.call_count, 1)


if __name__ == "__main__":
    unittest.main()
