import json
import os
import traceback
import threading
import tempfile
import time
from typing import Any
from pathlib import Path

from mAgents.pre_analysis_agent import create_pre_analysis_agent
from mAgents.data_analysis_agent import create_data_analysis_agent
from mAgents.model_agent import create_model_agent
from mAgents.ecg_tool_router import compose_routing_context, route_ecg_analysis
from mAgents.llm_checker import ECGQA_EMA_checker
from agent_reflect.skills.hierarchy import (
    PendingSkillGenerator,
    ReflectiveSkillGenerator,
    SkillComposer,
    SkillEvaluator,
    SkillExtractor,
    ReflectionGenerator,
    SkillRegistry,
    SkillRetriever,
    _merge_list,
)
from agent_reflect.vlm.slice_audit import SegmentSliceAuditor, capture_agent_variables
from thread_executor import AgentTask, AgentTaskResult
from utils.logger import get_logger
from utils.prompt import (
    get_pathology_inquiry_prompt,
    get_ecg_analysis_prompt,
    get_ecgSignals_doc_prompt,
    get_local_models,
)

logger = get_logger()
# 用于保护agent memory文件写入的锁
_agent_mem_lock = threading.Lock()
_skill_registry_lock = threading.Lock()


def _atomic_write_json(path: str, data) -> None:
    """
    原子写入JSON：先写到临时文件，再用os.replace替换，避免中途退出导致文件被截断/损坏。
    如果序列化失败，会先尝试清理数据再写入。
    """
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)

    fd = None
    tmp_path = None
    try:
        # 先尝试直接序列化
        try:
            json.dumps(data)
        except (TypeError, ValueError) as e:
            # 如果直接序列化失败，先清理数据
            logger.warning(f"Data not directly JSON serializable, cleaning first: {e}")
            data = _clean_for_json_serialization(data)
            if not _is_json_serializable(data):
                raise ValueError(f"Data still not serializable after cleaning: {type(data)}")
        
        fd, tmp_path = tempfile.mkstemp(prefix=".agent_mem_", suffix=".json", dir=dir_name)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        fd = None
        os.replace(tmp_path, path)
        tmp_path = None
    except Exception as e:
        logger.error(f"Failed to write JSON file {path}: {e}")
        raise
    finally:
        # best-effort cleanup
        try:
            if fd is not None:
                os.close(fd)
        except Exception:
            pass
        try:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _is_json_serializable(obj):
    """
    检查对象是否可JSON序列化
    
    Args:
        obj: 要检查的对象
    
    Returns:
        bool: 如果可序列化返回True，否则返回False
    """
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


def _clean_for_json_serialization(obj, visited=None):
    """
    递归清理嵌套dict/list，移除所有不可JSON序列化的内容
    
    Args:
        obj: 要清理的对象（dict, list, 或其他）
        visited: 用于防止循环引用的集合
    
    Returns:
        清理后的对象，所有不可序列化的内容已被移除
    """
    if visited is None:
        visited = set()
    
    # 防止循环引用
    obj_id = id(obj)
    if obj_id in visited:
        return None  # 遇到循环引用，返回None（会被移除）
    visited.add(obj_id)
    
    try:
        # 基本类型，直接检查是否可序列化
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        
        # 字典：递归清理每个值，移除不可序列化的key
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                # JSON的key必须是字符串，尝试转换
                try:
                    if not isinstance(key, str):
                        # 尝试转换为字符串
                        str_key = str(key)
                        if not _is_json_serializable(str_key):
                            continue  # key无法序列化，跳过
                    else:
                        str_key = key
                except Exception:
                    continue  # key转换失败，跳过
                
                # 递归清理value
                cleaned_value = _clean_for_json_serialization(value, visited)
                if cleaned_value is not None and _is_json_serializable(cleaned_value):
                    cleaned[str_key] = cleaned_value
                # 如果value不可序列化或清理后为None，则跳过该key
            return cleaned
        
        # 列表：递归清理每个元素，移除不可序列化的元素
        elif isinstance(obj, (list, tuple)):
            cleaned = []
            for item in obj:
                cleaned_item = _clean_for_json_serialization(item, visited)
                if cleaned_item is not None and _is_json_serializable(cleaned_item):
                    cleaned.append(cleaned_item)
            return cleaned
        
        # 其他类型：尝试转换为字符串，如果失败则返回None（会被移除）
        else:
            if _is_json_serializable(obj):
                return obj
            else:
                # 尝试转换为字符串表示
                try:
                    str_repr = str(obj)
                    if _is_json_serializable(str_repr):
                        return str_repr
                except Exception:
                    pass
                return None  # 无法序列化，返回None（会被移除）
    
    except Exception as e:
        logger.warning(f"Error cleaning object for JSON serialization: {e}")
        return None
    finally:
        visited.discard(obj_id)


def _get_agent_memory(agent):
    """
    获取agent的memory，尝试使用get_full_steps()方法，如果不存在则返回空列表

    Args:
        agent: CodeAgent实例

    Returns:
        list: agent的memory steps，如果无法获取则返回空列表
    """
    try:
        if hasattr(agent, 'memory') and hasattr(agent.memory, 'get_full_steps'):
            return agent.memory.get_full_steps()
        elif hasattr(agent, 'memory') and hasattr(agent.memory, 'steps'):
            # 如果get_full_steps不存在，尝试直接访问steps并转换为可序列化格式
            # 注意：这可能需要根据实际的step对象结构进行调整
            return []
    except Exception:
        pass
    return []


def _collect_agent_memories(*, model_agent, data_analysis_agent, synthesis_agent, pre_analysis_agent) -> dict[str, Any]:
    return {
        "model_agent": _get_agent_memory(model_agent),
        "data_analysis_agent": _get_agent_memory(data_analysis_agent),
        "synthesis_agent": _get_agent_memory(synthesis_agent),
        "pre_analysis_agent": _get_agent_memory(pre_analysis_agent),
    }


def _build_ecg_inputs(sample: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    ecgs_list = sample["ecg_datas"]
    if len(ecgs_list) == 1:
        return ecgs_list, {"ecg": ecgs_list[0]}

    ecgs = {f"ecg_{idx}": ecg for idx, ecg in enumerate(ecgs_list)}
    return ecgs_list, ecgs


def _infer_reflection_fs(ecgs: dict[str, Any], default_fs: int = 500) -> int:
    for ecg in ecgs.values():
        try:
            if hasattr(ecg, "get_fs"):
                fs = ecg.get_fs()
            else:
                fs = getattr(ecg, "fs", None)
            if fs:
                return int(fs)
        except Exception:
            continue
    return default_fs


def _run_model_agent(question: str, choices: list[str], ecgs_list: list[Any], model_agent) -> Any:
    """
    预留模型阶段入口。当前模型链路仍禁用，但保留明确阶段，后续恢复时只需要改这里。
    """
    del question, choices, ecgs_list, model_agent
    return None


def _run_pre_analysis_agent(
    question: str,
    choices: list[str],
    model_agent_res: Any,
    pre_analysis_agent,
    skill_context: str | None = None,
    local_pre_analysis_context: dict[str, Any] | None = None,
) -> Any:
    local_context = local_pre_analysis_context or {
        "plan_text": "No local pre-analysis context was prepared.",
        "remote_search_needed": True,
        "open_questions_for_remote_search": [],
    }
    if not local_context.get("remote_search_needed", True):
        return local_context.get("plan_text", "No local pre-analysis plan was produced.")

    pre_analysis_agent_prompt = get_pathology_inquiry_prompt(
        question,
        choices,
        model_agent_res,
        skill_context,
        local_pre_analysis_context=local_context.get("plan_text"),
        remote_search_needed=bool(local_context.get("remote_search_needed", True)),
        open_questions_for_remote_search=list(local_context.get("open_questions_for_remote_search", [])),
    )
    return pre_analysis_agent.run(pre_analysis_agent_prompt)


def _run_data_analysis_agent(
    question: str,
    choices: list[str],
    pre_analysis_agent_res: Any,
    ecgs: dict[str, Any],
    model_agent_res: Any,
    skill_context: str,
    routing_decision: dict[str, Any],
    data_analysis_agent,
) -> Any:
    routing_context = compose_routing_context(routing_decision)
    data_analysis_agent_prompt = get_ecg_analysis_prompt(
        question,
        choices,
        pre_analysis_agent_res,
        list(ecgs.keys()),
        model_agent_res,
        skill_context,
        routing_context,
    )
    data_analysis_agent.final_answer_checks = [ECGQA_EMA_checker(choices).check]
    data_analysis_agent.state.update(ecgs)
    data_analysis_agent.state.update(
        {
            "retrieved_micro_skills": skill_context,
            "preferred_ecg_tool": routing_decision.get("preferred_tool", "none"),
            "ecg_tool_routing_reason": routing_decision.get("reason", ""),
            "ecg_tool_constraints": routing_decision.get("routing_constraints", ""),
            "ecg_tool_routing_decision": routing_decision,
        }
    )
    return data_analysis_agent.run(data_analysis_agent_prompt)


def _build_result_entry(iteration: int, sample: dict[str, Any], final_res: Any) -> dict[str, Any]:
    sample_to_save = sample.copy()
    sample_to_save.pop("ecg_datas", None)
    sample_to_save["final_res"] = final_res
    sample_to_save["iteration"] = iteration
    return sample_to_save


def _get_skill_registry_path(base_path: str | None = None) -> str:
    default_storage_path = Path(__file__).resolve().parent / "agent_reflect" / "storage" / "agent_skill_registry.json"
    if base_path:
        base_parent = Path(base_path).resolve().parent
        if base_parent == Path(__file__).resolve().parent:
            return str(default_storage_path.resolve())
        return str((base_parent / "agent_reflect" / "storage" / "agent_skill_registry.json").resolve())
    return str(default_storage_path.resolve())


def _retrieve_skill_context(question: str, pre_analysis_agent_res: Any, registry_path: str) -> tuple[str, list[str], str]:
    registry = SkillRegistry(registry_path).load()
    retrieved_skills = SkillRetriever().retrieve(
        registry=registry,
        question=question,
        pre_analysis_res=None if pre_analysis_agent_res is None else str(pre_analysis_agent_res),
        top_k=5,
    )
    skill_context = SkillComposer().compose(retrieved_skills)
    return skill_context, [skill.skill_id for skill in retrieved_skills], registry.last_retrieval_mode


def _retrieve_pre_analysis_skill_context(
    question: str,
    registry_path: str,
) -> tuple[str, list[str], str]:
    registry = SkillRegistry(registry_path).load()
    retrieved_skills = SkillRetriever().retrieve(
        registry=registry,
        question=question,
        pre_analysis_res=None,
        top_k=5,
    )
    skill_context = SkillComposer().compose_pre_analysis(retrieved_skills)
    return skill_context, [skill.skill_id for skill in retrieved_skills], registry.last_retrieval_mode


def _empty_skill_context() -> tuple[str, list[str], str]:
    return "Relevant micro-skills:\n- None retrieved from prior tasks.", [], "disabled"


def _build_local_pre_analysis_context(
    *,
    question: str,
    choices: list[str],
    skill_context: str | None,
    routing_decision: dict[str, Any],
) -> dict[str, Any]:
    question_lower = question.lower()
    has_skills = bool(skill_context and "None retrieved from prior tasks" not in skill_context and "None retrieved from prior data_analysis_agent tasks" not in skill_context)
    has_conflicts = bool(skill_context and "conflicts" in skill_context.lower())
    open_questions: list[str] = []

    if not has_skills:
        open_questions.append("Identify which ECG features should be analyzed first and which local tool path is most clinically appropriate.")
    if has_conflicts:
        open_questions.append("Resolve conflicts between local skill workflows using medical diagnostic criteria or thresholds.")
    if any(token in question_lower for token in ["criteria", "threshold", "normal range", "diagnostic", "symptom", "symptoms", "identify", "explain"]):
        open_questions.append("Confirm ECG feature definitions, clinical thresholds, or diagnostic criteria relevant to the answer options.")
    if any(token in question_lower for token in ["which leads", "lead ", "lead i", "lead ii", "lead iii", "avl", "avr", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]):
        open_questions.append("Confirm lead-specific interpretation rules or expected lead distribution for the target feature.")
    if any(token in question_lower for token in ["qrs", "qt", "pr", "rr", "interval", "duration", "st", "t wave", "p wave", "morphology", "noise"]):
        open_questions.append("Fill any remaining clinical interpretation gaps for the target feature after local tool planning.")

    deduped_open_questions = _merge_list(open_questions, [])
    remote_search_needed = bool(deduped_open_questions)
    routing_summary = compose_routing_context(routing_decision)
    local_skill_summary = skill_context or "Local reusable skill plan:\n- None retrieved from prior tasks."
    tool_summary = get_local_models()
    ecg_api_summary = get_ecgSignals_doc_prompt()

    plan_lines = [
        "Local pre-analysis context:",
        "relevant_skills:",
        local_skill_summary,
        "tool_routing_summary:",
        routing_summary,
        "available_local_tools:",
        tool_summary.strip(),
        "available_local_ecg_api:",
        ecg_api_summary.strip(),
        "known_limits:",
        "- Local skill and routing guidance are capabilities, not facts about the current ECG.",
        "- Prefer local tool paths before remote search.",
        "open_questions_for_remote_search:",
    ]
    if deduped_open_questions:
        plan_lines.extend(f"- {item}" for item in deduped_open_questions)
    else:
        plan_lines.append("- None. Local skill and tool guidance already cover this task.")
    plan_lines.append(f"final_search_needed: {'yes' if remote_search_needed else 'no'}")
    plan_lines.append(f"candidate_answers_count: {len(choices)}")
    return {
        "plan_text": "\n".join(plan_lines),
        "remote_search_needed": remote_search_needed,
        "open_questions_for_remote_search": deduped_open_questions,
    }


def _combine_skill_context(base_context: str, retry_context: str) -> str:
    if not retry_context:
        return base_context
    return "\n\n".join(
        [
            base_context,
            "Retry-only pending skills:",
            retry_context,
            "Retry constraints:",
            "- These pending skills are temporary and unverified; use them to re-check the reasoning path, not as facts.",
            "- Do not use or reveal the gold answer. Re-derive the answer from ECG evidence and allowed choices only.",
        ]
    )


def _extract_and_store_micro_skills(
    *,
    sample: dict[str, Any],
    question: str,
    choices: list[str],
    data_analysis_memory: list[Any],
    final_answer: str | None,
    registry_path: str,
    retrieved_skill_ids: list[str],
) -> list[str]:
    extractor = SkillExtractor()
    evaluator = SkillEvaluator()
    candidates = extractor.extract(
        sample_id=str(sample.get("sample_id", "")),
        question_id=str(sample.get("question_id", "")),
        template_id=str(sample.get("template_id", "")),
        question=question,
        choices=choices,
        data_analysis_memory=data_analysis_memory,
        final_answer=final_answer,
    )
    with _skill_registry_lock:
        registry = SkillRegistry(registry_path).load()
        accepted, _rejected = evaluator.evaluate(candidates, registry)
        registry.increment_reuse(retrieved_skill_ids)
        accepted_ids = [skill.skill_id for skill in accepted]
        registry.save()
    return accepted_ids


def _generate_reflective_skills(
    *,
    sample: dict[str, Any],
    question: str,
    choices: list[str],
    data_analysis_memory: list[Any],
    final_answer: str | None,
    predicted_answer: str | None = None,
    gold_answers: list[str] | None = None,
    mode: str = "success",
) -> list[Any]:
    generator = ReflectiveSkillGenerator()
    return generator.generate(
        sample_id=str(sample.get("sample_id", "")),
        question_id=str(sample.get("question_id", "")),
        template_id=str(sample.get("template_id", "")),
        question=question,
        choices=choices,
        data_analysis_memory=data_analysis_memory,
        final_answer=final_answer,
        predicted_answer=predicted_answer,
        gold_answers=gold_answers or [],
        mode=mode,
    )


def _store_reflective_skills(
    *,
    reflective_skills: list[Any],
    registry_path: str,
    retrieved_skill_ids: list[str],
) -> list[str]:
    evaluator = SkillEvaluator()
    with _skill_registry_lock:
        registry = SkillRegistry(registry_path).load()
        accepted, _rejected = evaluator.evaluate(reflective_skills, registry)
        registry.increment_reuse(retrieved_skill_ids)
        accepted_ids = [skill.skill_id for skill in accepted if getattr(skill, "status", "active") == "active"]
        registry.save()
    return accepted_ids


def _is_prediction_correct(predicted_answer: Any, gold_answers: list[str]) -> bool:
    if not isinstance(predicted_answer, str):
        return False
    normalized_prediction = predicted_answer.strip().lower()
    normalized_gold = {answer.strip().lower() for answer in gold_answers if isinstance(answer, str)}
    return normalized_prediction in normalized_gold


def _load_all_memories(agent_mem_file: str, sample_id: Any) -> dict[str, Any]:
    if not os.path.exists(agent_mem_file):
        return {}

    try:
        with open(agent_mem_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to load agent memory for sample_id {sample_id}: {e}")
        logger.error(traceback.format_exc())
        try:
            ts = time.strftime("%Y%m%d-%H%M%S")
            backup_path = f"{agent_mem_file}.corrupt-{ts}"
            os.replace(agent_mem_file, backup_path)
        except Exception:
            pass
        return {}
    except IOError:
        return {}


def _save_agent_memories(
    agent_mem_file: str,
    agent_memories: dict[str, Any],
    sample_id: Any,
) -> None:
    if sample_id is None:
        return

    agent_memories = _clean_for_json_serialization(agent_memories)
    if not isinstance(agent_memories, dict):
        logger.warning(f"Cleaned agent_memories is not a dict for sample_id {sample_id}, skipping save")
        agent_memories = {}

    with _agent_mem_lock:
        all_memories = _load_all_memories(agent_mem_file, sample_id)
        cleaned_memories = _clean_for_json_serialization(agent_memories)
        if isinstance(cleaned_memories, dict):
            all_memories[str(sample_id)] = cleaned_memories
        else:
            logger.warning(f"Cleaned memories for sample_id {sample_id} is not a dict, using empty dict")
            all_memories[str(sample_id)] = {}

        all_memories = _clean_for_json_serialization(all_memories)
        if not isinstance(all_memories, dict):
            logger.error("Failed to clean all_memories, using empty dict")
            all_memories = {}

        _atomic_write_json(agent_mem_file, all_memories)


def process_sample(task: AgentTask) -> AgentTaskResult:
    """
    处理单个样本，执行固定的 agent pipeline:
    model -> pre_analysis -> data_analysis

    Args:
        task: 线程执行器分发的任务对象

    Returns:
        AgentTaskResult: 线程层消费的标准结果结构
    """
    try:
        iter_idx = task.iteration
        sample = task.sample

        pre_analysis_agent = create_pre_analysis_agent()
        data_analysis_agent = create_data_analysis_agent()
        model_agent = create_model_agent()

        question = sample["question"]
        choices = sample["template_answer"]
        gold_answers = sample.get("answer", [])
        ecgs_list, ecgs = _build_ecg_inputs(sample)
        registry_path = _get_skill_registry_path(task.options.agent_mem_file)
        run_mode = getattr(task.options, "run_mode", "eval")
        skill_build_mode = run_mode == "skill_build"

        model_agent_res = _run_model_agent(question, choices, ecgs_list, model_agent)
        initial_routing_decision = route_ecg_analysis(
            question=question,
            choices=choices,
            pre_analysis_res=None,
            ecgs=ecgs,
        )
        if skill_build_mode:
            pre_search_skill_context, pre_search_skill_ids, pre_search_retrieval_mode = _retrieve_pre_analysis_skill_context(
                question,
                registry_path,
            )
        else:
            pre_search_skill_context, pre_search_skill_ids, pre_search_retrieval_mode = _empty_skill_context()
        local_pre_analysis_context = _build_local_pre_analysis_context(
            question=question,
            choices=choices,
            skill_context=pre_search_skill_context if pre_search_skill_ids else None,
            routing_decision=initial_routing_decision,
        )
        pre_analysis_agent_res = _run_pre_analysis_agent(
            question,
            choices,
            model_agent_res,
            pre_analysis_agent,
            pre_search_skill_context if pre_search_skill_ids else None,
            local_pre_analysis_context=local_pre_analysis_context,
        )
        if skill_build_mode:
            skill_context, retrieved_skill_ids, data_skill_retrieval_mode = _retrieve_skill_context(
                question,
                pre_analysis_agent_res,
                registry_path,
            )
        else:
            skill_context, retrieved_skill_ids, data_skill_retrieval_mode = _empty_skill_context()
        routing_decision = route_ecg_analysis(
            question=question,
            choices=choices,
            pre_analysis_res=None if pre_analysis_agent_res is None else str(pre_analysis_agent_res),
            ecgs=ecgs,
        )
        data_analysis_agent_res = _run_data_analysis_agent(
            question,
            choices,
            pre_analysis_agent_res,
            ecgs,
            model_agent_res,
            skill_context,
            routing_decision,
            data_analysis_agent,
        )
        final_res = data_analysis_agent_res
        is_correct = _is_prediction_correct(final_res, gold_answers)
        first_final_res = final_res
        retry_reflection: dict[str, str] | None = None
        retry_used = False
        vlm_slice_audit: dict[str, Any] | None = None
        data_analysis_variable_summaries: list[dict[str, Any]] = []
        reflection_skipped = False
        reflection_skip_reason = ""
        reflective_retry_skills: list[Any] = []

        first_data_analysis_memory: list[Any] = []
        if skill_build_mode and not is_correct:
            first_data_analysis_memory = _get_agent_memory(data_analysis_agent)
            reflection_fs = _infer_reflection_fs(ecgs)
            variable_capture = capture_agent_variables(
                data_analysis_agent,
                first_data_analysis_memory,
                fs=reflection_fs,
            )
            data_analysis_variable_summaries = variable_capture.get("summaries", [])
            vlm_slice_audit = SegmentSliceAuditor(fs=reflection_fs).audit(
                question=question,
                choices=choices,
                variable_capture=variable_capture,
                data_analysis_memory=first_data_analysis_memory,
            )

            if vlm_slice_audit.get("verdict") == "segmentation_error":
                reflection_skipped = True
                reflection_skip_reason = "segmentation_error"
            elif vlm_slice_audit.get("verdict") == "invalid_slice":
                reflection_skipped = True
                reflection_skip_reason = "invalid_slice"
            else:
                retry_reflection = ReflectionGenerator().generate(
                    question=question,
                    predicted_answer=None if final_res is None else str(final_res),
                    gold_answers=gold_answers,
                    is_correct=False,
                    data_analysis_memory=first_data_analysis_memory,
                    used_skill_names=[],
                    learned_skill_names=[],
                    variable_summaries=data_analysis_variable_summaries,
                    vlm_slice_audit=vlm_slice_audit,
                )
                reflective_retry_skills = _generate_reflective_skills(
                    sample=sample,
                    question=question,
                    choices=choices,
                    data_analysis_memory=first_data_analysis_memory,
                    final_answer=None,
                    predicted_answer=None if final_res is None else str(final_res),
                    gold_answers=gold_answers,
                    mode="failure_retry",
                )
                pending_skill_context = PendingSkillGenerator().compose_retry_context(
                    reflective_retry_skills,
                    None if final_res is None else str(final_res),
                )
                retry_data_analysis_agent = create_data_analysis_agent()
                retry_context = _combine_skill_context(skill_context, pending_skill_context)
                retry_final_res = _run_data_analysis_agent(
                    question,
                    choices,
                    pre_analysis_agent_res,
                    ecgs,
                    model_agent_res,
                    retry_context,
                    routing_decision,
                    retry_data_analysis_agent,
                )
                retry_used = True
                data_analysis_agent = retry_data_analysis_agent
                final_res = retry_final_res
                is_correct = _is_prediction_correct(final_res, gold_answers)

        result_entry = _build_result_entry(iter_idx, sample, final_res)
        result_entry["pre_analysis_skill_gate"] = bool(pre_search_skill_ids)
        result_entry["pre_analysis_gate_skills"] = pre_search_skill_ids
        result_entry["pre_analysis_skill_retrieval_mode"] = pre_search_retrieval_mode
        result_entry["pre_analysis_remote_search_needed"] = bool(local_pre_analysis_context.get("remote_search_needed", True))
        result_entry["pre_analysis_open_questions"] = local_pre_analysis_context.get("open_questions_for_remote_search", [])
        result_entry["used_data_analysis_skills"] = retrieved_skill_ids
        result_entry["data_analysis_skill_retrieval_mode"] = data_skill_retrieval_mode
        result_entry["ecg_tool_routing_decision"] = routing_decision
        result_entry["is_correct"] = is_correct
        result_entry["retry_used"] = retry_used
        result_entry["run_mode"] = run_mode
        if vlm_slice_audit is not None:
            result_entry["vlm_slice_audit"] = vlm_slice_audit
            result_entry["data_analysis_variable_summaries"] = data_analysis_variable_summaries
        if reflection_skipped:
            result_entry["reflection_skipped"] = True
            result_entry["reflection_skip_reason"] = reflection_skip_reason
        if retry_used:
            result_entry["first_final_res"] = first_final_res
            result_entry["retry_reflection"] = retry_reflection

        agent_memories = _collect_agent_memories(
            model_agent=model_agent,
            data_analysis_agent=data_analysis_agent,
            synthesis_agent=None,
            pre_analysis_agent=pre_analysis_agent,
        )
        if retry_used:
            agent_memories["first_data_analysis_agent"] = first_data_analysis_memory
        if data_analysis_variable_summaries:
            agent_memories["data_analysis_variable_summaries"] = data_analysis_variable_summaries
        if vlm_slice_audit is not None:
            agent_memories["vlm_slice_audit"] = vlm_slice_audit
        learned_skill_ids: list[str] = []

        if task.options.save_agent_mem:
            try:
                _save_agent_memories(
                    task.options.agent_mem_file,
                    agent_memories,
                    sample.get("sample_id"),
                )
            except Exception as e:
                print(f"Warning: Failed to save agent memory for sample_id {sample.get('sample_id')}: {e}")

        learned_skill_names: list[str] = []
        if skill_build_mode and not reflection_skipped and is_correct:
            try:
                reflective_success_skills = _generate_reflective_skills(
                    sample=sample,
                    question=question,
                    choices=choices,
                    data_analysis_memory=agent_memories.get("data_analysis_agent", []),
                    final_answer=None if final_res is None else str(final_res),
                    mode="success",
                )
                learned_skill_ids = _store_reflective_skills(
                    reflective_skills=reflective_success_skills,
                    registry_path=registry_path,
                    retrieved_skill_ids=retrieved_skill_ids,
                )
            except Exception as e:
                print(f"Warning: Failed to update reflective skill registry for sample_id {sample.get('sample_id')}: {e}")

        if skill_build_mode and retry_used and is_correct and reflective_retry_skills:
            try:
                for skill in reflective_retry_skills:
                    skill.status = "active"
                    skill.applies_on_retry_only = False
                    skill.validation_basis = ["retry_success"]
                retry_skill_ids = _store_reflective_skills(
                    reflective_skills=reflective_retry_skills,
                    registry_path=registry_path,
                    retrieved_skill_ids=retrieved_skill_ids,
                )
                learned_skill_ids = _merge_list(learned_skill_ids, retry_skill_ids)
            except Exception as e:
                print(f"Warning: Failed to promote retry skills for sample_id {sample.get('sample_id')}: {e}")

        if skill_build_mode and retry_used and not is_correct and reflective_retry_skills:
            try:
                with _skill_registry_lock:
                    registry = SkillRegistry(registry_path).load()
                    for skill in reflective_retry_skills:
                        skill.status = "pending_review"
                        skill.applies_on_retry_only = False
                        registry.upsert_skill(skill)
                    registry.save()
            except Exception as e:
                print(f"Warning: Failed to record pending retry skills for sample_id {sample.get('sample_id')}: {e}")

        if skill_build_mode and not reflection_skipped:
            try:
                with _skill_registry_lock:
                    registry = SkillRegistry(registry_path).load()
                    used_skill_names = [skill.name for skill in registry.skills if skill.skill_id in retrieved_skill_ids]
                    learned_skill_names = [skill.name for skill in registry.skills if skill.skill_id in learned_skill_ids]
                    reflection = ReflectionGenerator().generate(
                        question=question,
                        predicted_answer=None if final_res is None else str(final_res),
                        gold_answers=gold_answers,
                        is_correct=is_correct,
                        data_analysis_memory=agent_memories.get("data_analysis_agent", []),
                        used_skill_names=used_skill_names,
                        learned_skill_names=learned_skill_names,
                        variable_summaries=data_analysis_variable_summaries,
                        vlm_slice_audit=vlm_slice_audit,
                    )
                    registry.record_outcome(
                        retrieved_skill_ids + learned_skill_ids,
                        success=is_correct,
                        reflection=reflection["summary"] + " " + reflection["root_cause"],
                    )
                    registry.save()
                result_entry["reflection"] = reflection
            except Exception as e:
                print(f"Warning: Failed to generate reflection for sample_id {sample.get('sample_id')}: {e}")

        result_entry["learned_data_analysis_skills"] = learned_skill_ids
        return AgentTaskResult(iteration=iter_idx, result_entry=result_entry)
    except Exception:
        return AgentTaskResult(
            iteration=task.iteration,
            result_entry=None,
            error=traceback.format_exc(),
        )
