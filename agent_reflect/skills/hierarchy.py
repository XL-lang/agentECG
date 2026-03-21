import csv
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_+\-]+", text.lower())
        if len(token) > 2
    }


def collect_texts(node: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(node, dict):
        for value in node.values():
            texts.extend(collect_texts(value))
    elif isinstance(node, list):
        for item in node:
            texts.extend(collect_texts(item))
    elif isinstance(node, str):
        texts.append(node)
    return texts


_NATIVE_CAPABILITY_CATALOG: dict[str, dict[str, tuple[str, ...]]] = {
    "ecgdeli_structured_measurement": {
        "feature_scopes": (
            "pr_interval",
            "qt_interval",
            "qrs_duration",
            "rr_interval",
            "r_amplitude",
        ),
        "tool_dependencies": ("ECGDeli_measurement_tool",),
        "action_markers": (
            "consume ecgdeli feature-level measurement output",
            "verify sampling frequency and convert sample span into duration",
            "slice the narrowest ecg segment associated with the feature",
        ),
    },
    "ecg_api_waveform_access": {
        "feature_scopes": (),
        "tool_dependencies": (),
        "action_markers": (
            "read target lead waveform before drawing a feature conclusion",
            "slice the narrowest ecg segment associated with the feature",
        ),
    },
    "visual_tool_native_inspection": {
        "feature_scopes": (),
        "tool_dependencies": ("ECG_fig_anaysis_tool",),
        "action_markers": (
            "route morphology-sensitive verification to the visual ecg tool",
        ),
    },
}

_GENERIC_NATIVE_ACTION_PREFIXES = (
    "read target lead waveform before drawing a feature conclusion",
    "slice the narrowest ecg segment associated with the feature",
    "consume ecgdeli feature-level measurement output for the target feature",
    "verify sampling frequency and convert sample span into duration only after confirmation",
    "route morphology-sensitive verification to the visual ecg tool",
    "map validated `",
)

_EXPERIENCE_SIGNAL_MARKERS = (
    "avoid",
    "prefer not",
    "do not",
    "fallback",
    "over-trusted",
    "misleading",
    "over-weighted",
    "false positive",
    "false-negative",
    "shrink evidence window",
    "narrow validation before selecting an answer",
)


def _matches_native_action(action: str) -> bool:
    lowered = _normalize(action)
    return any(lowered.startswith(prefix) for prefix in _GENERIC_NATIVE_ACTION_PREFIXES)


def _has_experience_signal(skill: "SkillUnit") -> bool:
    joined = " ".join(
        [
            skill.description,
            skill.procedure_summary,
            skill.applicability,
            " ".join(skill.failure_modes),
            " ".join(skill.reflection_notes),
        ]
    ).lower()
    return any(marker in joined for marker in _EXPERIENCE_SIGNAL_MARKERS)


def _native_capability_family(skill: "SkillUnit") -> str:
    feature_scope = _normalize(skill.feature_scope)
    tool_dependencies = {_normalize(tool) for tool in skill.tool_dependencies}
    action_bundle = [_normalize(action) for action in skill.action_bundle]

    structured_payload = _NATIVE_CAPABILITY_CATALOG["ecgdeli_structured_measurement"]
    structured_scopes = {_normalize(item) for item in structured_payload["feature_scopes"]}
    structured_tools = {_normalize(item) for item in structured_payload["tool_dependencies"]}
    structured_actions = {_normalize(item) for item in structured_payload["action_markers"]}
    if feature_scope and feature_scope in structured_scopes:
        return "ecgdeli_structured_measurement"
    if tool_dependencies & structured_tools:
        if any(any(marker in action for marker in structured_actions) for action in action_bundle):
            return "ecgdeli_structured_measurement"

    if feature_scope in {"", "general_ecg_feature"}:
        api_actions = {_normalize(item) for item in _NATIVE_CAPABILITY_CATALOG["ecg_api_waveform_access"]["action_markers"]}
        if any(any(marker in action for marker in api_actions) for action in action_bundle):
            return "ecg_api_waveform_access"

        visual_tools = {_normalize(item) for item in _NATIVE_CAPABILITY_CATALOG["visual_tool_native_inspection"]["tool_dependencies"]}
        visual_actions = {_normalize(item) for item in _NATIVE_CAPABILITY_CATALOG["visual_tool_native_inspection"]["action_markers"]}
        if tool_dependencies & visual_tools:
            if any(any(marker in action for marker in visual_actions) for action in action_bundle):
                return "visual_tool_native_inspection"

    return ""


def _is_native_capability_duplicate(skill: "SkillUnit") -> bool:
    family = _native_capability_family(skill)
    if not family:
        return False

    if _has_experience_signal(skill):
        return False

    if not skill.action_bundle:
        return True

    native_actions = sum(1 for action in skill.action_bundle if _matches_native_action(action))
    return native_actions >= max(2, len(skill.action_bundle) - 1)


@dataclass
class SkillUnit:
    skill_id: str
    agent_name: str
    name: str
    description: str
    level: str
    category: str
    input_requirements: list[str]
    procedure_summary: str
    evidence_pattern: list[str]
    tool_dependencies: list[str]
    applicability: str
    failure_modes: list[str]
    source_sample_ids: list[str]
    confidence: float
    reuse_count: int
    created_at: str
    updated_at: str
    success_count: int = 0
    failure_count: int = 0
    reflection_notes: list[str] = field(default_factory=list)
    last_reflection: str = ""
    family_id: str = ""
    conflict_with: list[str] = field(default_factory=list)
    source_question_ids: list[str] = field(default_factory=list)
    source_task_refs: list[dict[str, str]] = field(default_factory=list)
    source_trace_file: str = ""
    skill_source: str = "success_trace"
    feature_scope: str = ""
    action_bundle: list[str] = field(default_factory=list)
    validation_basis: list[str] = field(default_factory=list)
    status: str = "active"
    origin_prediction: str = ""
    origin_gold: list[str] = field(default_factory=list)
    applies_on_retry_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillUnit":
        merged = {
            "success_count": 0,
            "failure_count": 0,
            "reflection_notes": [],
            "last_reflection": "",
            "family_id": "",
            "conflict_with": [],
            "source_question_ids": [],
            "source_task_refs": [],
            "source_trace_file": "",
            "skill_source": "success_trace",
            "feature_scope": "",
            "action_bundle": [],
            "validation_basis": [],
            "status": "active",
            "origin_prediction": "",
            "origin_gold": [],
            "applies_on_retry_only": False,
            **data,
        }
        return cls(**merged)


class SkillRegistry:
    def __init__(self, path: str | Path, enable_semantic_merge: bool = True):
        self.path = Path(path)
        self.source_trace_path = self.path.with_name("agent_skill_sources.jsonl")
        self.conflict_table_path = self.path.with_name("agent_skill_conflicts.csv")
        self.pending_review_path = self.path.with_name("agent_skill_pending_reviews.json")
        self.skills: list[SkillUnit] = []
        self.last_retrieval_mode = "none"
        self.semantic_merger = SkillSemanticMerger(enabled=enable_semantic_merge)

    def load(self) -> "SkillRegistry":
        if not self.path.exists():
            self.skills = []
            return self

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.skills = []
            return self

        self.skills = [SkillUnit.from_dict(item) for item in raw.get("skills", [])]
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "skills": [skill.to_dict() for skill in self.skills],
        }
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def upsert_skill(self, skill: SkillUnit) -> SkillUnit:
        if not skill.family_id:
            skill.family_id = build_family_id(skill)
        skill.source_trace_file = str(self.source_trace_path)

        if skill.status == "pending_review":
            self._store_pending_review(skill, reason="explicit pending review")
            return skill

        existing = self._find_by_signature(skill)
        if existing is not None:
            self._merge_skill(existing, skill)
            self.record_source_trace(existing, skill)
            return existing

        semantic_relation = self.semantic_merger.find_semantic_relation(skill, self.skills)
        if semantic_relation is None:
            self.skills.append(skill)
            self.record_source_trace(skill, skill)
            return skill

        existing_skill, relation_payload = semantic_relation
        if relation_payload.get("decision") == "conflict":
            self._record_conflicting_skill(existing_skill, skill, relation_payload)
            return skill

        self._merge_skill(existing_skill, skill, relation_payload)
        self.record_source_trace(existing_skill, skill)
        return existing_skill

    def find_relevant_skills(
        self,
        question: str,
        pre_analysis_res: str | None = None,
        top_k: int = 5,
    ) -> list[SkillUnit]:
        llm_selected = SkillRetriever().retrieve_with_llm(
            skills=self.skills,
            question=question,
            pre_analysis_res=pre_analysis_res,
            top_k=top_k,
        )
        if llm_selected:
            self.last_retrieval_mode = "llm"
            return llm_selected
        self.last_retrieval_mode = "overlap"
        return self.find_relevant_skills_by_overlap(question, pre_analysis_res, top_k)

    def find_relevant_skills_by_overlap(
        self,
        question: str,
        pre_analysis_res: str | None = None,
        top_k: int = 5,
    ) -> list[SkillUnit]:
        query_tokens = _tokenize(f"{question} {pre_analysis_res or ''}")
        scored: list[tuple[int, int, SkillUnit]] = []

        for skill in self.skills:
            if skill.status != "active" or skill.applies_on_retry_only:
                continue
            haystack = " ".join(
                [
                    skill.name,
                    skill.description,
                    skill.applicability,
                    skill.procedure_summary,
                    skill.feature_scope,
                    " ".join(skill.action_bundle),
                    " ".join(skill.evidence_pattern),
                    " ".join(skill.failure_modes),
                ]
            )
            overlap = len(query_tokens & _tokenize(haystack))
            if overlap == 0:
                continue
            scored.append((overlap, skill.reuse_count, skill))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:top_k]]

    def increment_reuse(self, skill_ids: list[str]) -> None:
        if not skill_ids:
            return
        for skill in self.skills:
            if skill.skill_id in skill_ids:
                skill.reuse_count += 1
                skill.updated_at = _utc_now()

    def record_outcome(self, skill_ids: list[str], success: bool, reflection: str = "") -> None:
        if not skill_ids:
            return
        delta = 0.02 if success else -0.05
        for skill in self.skills:
            if skill.skill_id in skill_ids:
                skill.confidence = max(0.0, min(1.0, skill.confidence + delta))
                if success:
                    skill.success_count += 1
                else:
                    skill.failure_count += 1
                if reflection:
                    skill.last_reflection = reflection
                    if reflection not in skill.reflection_notes:
                        skill.reflection_notes.append(reflection)
                        skill.reflection_notes = skill.reflection_notes[-5:]
                skill.updated_at = _utc_now()

    def _find_by_signature(self, candidate: SkillUnit) -> SkillUnit | None:
        signature = _signature_for_skill(candidate)
        for skill in self.skills:
            if skill.status != "active":
                continue
            if _signature_for_skill(skill) == signature:
                return skill
        return None

    def _merge_skill(
        self,
        existing: SkillUnit,
        candidate: SkillUnit,
        semantic_payload: dict[str, Any] | None = None,
    ) -> None:
        payload = semantic_payload or {}
        existing.name = payload.get("merged_name") or existing.name
        existing.description = payload.get("merged_description") or existing.description
        existing.procedure_summary = payload.get("merged_procedure_summary") or existing.procedure_summary
        existing.applicability = payload.get("merged_applicability") or existing.applicability

        existing.input_requirements = _merge_list(existing.input_requirements, candidate.input_requirements)
        existing.evidence_pattern = _merge_list(existing.evidence_pattern, candidate.evidence_pattern)
        existing.tool_dependencies = _merge_list(existing.tool_dependencies, candidate.tool_dependencies)
        existing.failure_modes = _merge_list(
            existing.failure_modes,
            payload.get("merged_failure_modes") or candidate.failure_modes,
        )
        existing.action_bundle = _merge_list(existing.action_bundle, candidate.action_bundle)
        existing.validation_basis = _merge_list(existing.validation_basis, candidate.validation_basis)
        existing.source_sample_ids = _merge_list(existing.source_sample_ids, candidate.source_sample_ids)
        existing.source_question_ids = _merge_list(existing.source_question_ids, candidate.source_question_ids)
        existing.source_task_refs = _merge_task_refs(existing.source_task_refs, candidate.source_task_refs)
        existing.conflict_with = _merge_list(existing.conflict_with, candidate.conflict_with)
        existing.reflection_notes = _merge_list(existing.reflection_notes, candidate.reflection_notes)[-5:]
        existing.feature_scope = payload.get("merged_feature_scope") or existing.feature_scope or candidate.feature_scope
        existing.skill_source = existing.skill_source if existing.skill_source == candidate.skill_source else "success_trace"
        existing.status = "active"
        existing.origin_prediction = existing.origin_prediction or candidate.origin_prediction
        existing.origin_gold = _merge_list(existing.origin_gold, candidate.origin_gold)
        existing.applies_on_retry_only = False
        if not existing.family_id:
            existing.family_id = candidate.family_id or build_family_id(existing)
        existing.source_trace_file = str(self.source_trace_path)

        existing.confidence = max(existing.confidence, candidate.confidence)
        existing.success_count += candidate.success_count
        existing.failure_count += candidate.failure_count
        if candidate.last_reflection:
            existing.last_reflection = candidate.last_reflection
        if payload.get("reason"):
            note = f"semantic_merge: {payload['reason']}"
            if note not in existing.reflection_notes:
                existing.reflection_notes.append(note)
                existing.reflection_notes = existing.reflection_notes[-5:]
        existing.updated_at = _utc_now()

    def _record_conflicting_skill(
        self,
        existing: SkillUnit,
        candidate: SkillUnit,
        payload: dict[str, Any],
    ) -> None:
        if not existing.family_id:
            existing.family_id = build_family_id(existing)
        candidate.family_id = existing.family_id
        candidate.source_trace_file = str(self.source_trace_path)
        candidate.status = "pending_review"
        existing.conflict_with = _merge_list(existing.conflict_with, [candidate.skill_id])
        candidate.conflict_with = _merge_list(candidate.conflict_with, [existing.skill_id])

        note = f"conflict_with:{existing.skill_id}: {payload.get('reason', 'semantic conflict')}"
        if note not in candidate.reflection_notes:
            candidate.reflection_notes.append(note)
        existing_note = f"conflict_with:{candidate.skill_id}: {payload.get('reason', 'semantic conflict')}"
        if existing_note not in existing.reflection_notes:
            existing.reflection_notes.append(existing_note)
            existing.reflection_notes = existing.reflection_notes[-5:]

        self.skills.append(candidate)
        self.record_source_trace(candidate, candidate)
        self.record_conflict(existing, candidate, payload.get("reason", "semantic conflict"))
        self._store_pending_review(candidate, payload.get("reason", "semantic conflict"), existing)

    def record_source_trace(self, stored_skill: SkillUnit, source_skill: SkillUnit) -> None:
        trace = {
            "created_at": _utc_now(),
            "stored_skill_id": stored_skill.skill_id,
            "source_skill_id": source_skill.skill_id,
            "family_id": stored_skill.family_id or source_skill.family_id,
            "agent_name": source_skill.agent_name,
            "category": source_skill.category,
            "source_sample_ids": source_skill.source_sample_ids,
            "source_question_ids": source_skill.source_question_ids,
            "source_task_refs": source_skill.source_task_refs,
            "skill_name": source_skill.name,
            "procedure_summary": source_skill.procedure_summary,
        }
        self.source_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.source_trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    def record_conflict(self, existing: SkillUnit, candidate: SkillUnit, reason: str) -> None:
        self.conflict_table_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "created_at",
            "status",
            "family_id",
            "existing_skill_id",
            "candidate_skill_id",
            "category",
            "reason",
            "existing_name",
            "candidate_name",
            "existing_source_question_ids",
            "candidate_source_question_ids",
            "existing_source_task_refs",
            "candidate_source_task_refs",
        ]
        write_header = not self.conflict_table_path.exists()
        with self.conflict_table_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "created_at": _utc_now(),
                    "status": "pending",
                    "family_id": existing.family_id,
                    "existing_skill_id": existing.skill_id,
                    "candidate_skill_id": candidate.skill_id,
                    "category": candidate.category,
                    "reason": reason,
                    "existing_name": existing.name,
                    "candidate_name": candidate.name,
                    "existing_source_question_ids": ";".join(existing.source_question_ids),
                    "candidate_source_question_ids": ";".join(candidate.source_question_ids),
                    "existing_source_task_refs": json.dumps(existing.source_task_refs, ensure_ascii=False),
                    "candidate_source_task_refs": json.dumps(candidate.source_task_refs, ensure_ascii=False),
                }
            )

    def _store_pending_review(
        self,
        candidate: SkillUnit,
        reason: str,
        existing: SkillUnit | None = None,
    ) -> None:
        payload: list[dict[str, Any]]
        if self.pending_review_path.exists():
            try:
                payload = json.loads(self.pending_review_path.read_text(encoding="utf-8"))
                if not isinstance(payload, list):
                    payload = []
            except json.JSONDecodeError:
                payload = []
        else:
            payload = []

        payload.append(
            {
                "created_at": _utc_now(),
                "reason": reason,
                "feature_scope": candidate.feature_scope,
                "skill_source": candidate.skill_source,
                "status": candidate.status,
                "candidate": candidate.to_dict(),
                "existing": existing.to_dict() if existing is not None else None,
                "source_question_ids": candidate.source_question_ids,
                "source_task_refs": candidate.source_task_refs,
                "from_failure_retry": candidate.skill_source == "failure_retry",
            }
        )
        self.pending_review_path.parent.mkdir(parents=True, exist_ok=True)
        self.pending_review_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _merge_list(left: list[str], right: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*left, *right]:
        normalized = _normalize(str(item))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(str(item))
    return merged


def _merge_task_refs(
    left: list[dict[str, str]],
    right: list[dict[str, str]],
) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in [*left, *right]:
        if not isinstance(item, dict):
            continue
        normalized = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        merged.append(item)
    return merged


class SkillSemanticMerger:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and os.environ.get("AGENT_ECG_DISABLE_LLM_SKILL_MERGE") != "1"
        self.embedding_model = os.environ.get("AGENT_ECG_SKILL_EMBEDDING_MODEL", "").strip()

    def find_semantic_relation(
        self,
        candidate: SkillUnit,
        existing_skills: list[SkillUnit],
    ) -> tuple[SkillUnit, dict[str, Any]] | None:
        candidates = self._rank_local_candidates(candidate, existing_skills)
        if not candidates:
            return None

        for existing, score in candidates:
            if score >= 0.82:
                return existing, {
                    "decision": "merge",
                    "reason": f"high local semantic overlap ({score:.2f})",
                }
            if not self.enabled:
                continue
            decision = self._judge_with_llm(existing, candidate, score)
            if decision.get("decision") in {"merge", "conflict"}:
                return existing, decision

        return None

    def _rank_local_candidates(
        self,
        candidate: SkillUnit,
        existing_skills: list[SkillUnit],
    ) -> list[tuple[SkillUnit, float]]:
        scored: list[tuple[SkillUnit, float]] = []
        candidate_text = _skill_semantic_text(candidate)
        candidate_tokens = _tokenize(candidate_text)

        for skill in existing_skills:
            if skill.agent_name != candidate.agent_name:
                continue
            if skill.level != candidate.level:
                continue
            if skill.category != candidate.category:
                continue
            if skill.status != "active":
                continue
            if candidate.feature_scope and skill.feature_scope and _normalize(skill.feature_scope) != _normalize(candidate.feature_scope):
                continue

            skill_text = _skill_semantic_text(skill)
            skill_tokens = _tokenize(skill_text)
            token_score = _jaccard(candidate_tokens, skill_tokens)
            embedding_score = self._embedding_similarity(candidate_text, skill_text)
            tool_score = 0.2 if set(candidate.tool_dependencies) & set(skill.tool_dependencies) else 0.0
            evidence_score = 0.15 if set(map(_normalize, candidate.evidence_pattern)) & set(map(_normalize, skill.evidence_pattern)) else 0.0
            semantic_score = max(token_score, embedding_score or 0.0)
            total = min(1.0, semantic_score + tool_score + evidence_score)
            if total >= 0.18:
                scored.append((skill, total))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:3]

    def _embedding_similarity(self, left: str, right: str) -> float | None:
        if not self.embedding_model:
            return None
        try:
            from utils import reason_client

            response = reason_client.embeddings.create(
                model=self.embedding_model,
                input=[left, right],
            )
            left_vector = response.data[0].embedding
            right_vector = response.data[1].embedding
            return _cosine_similarity(left_vector, right_vector)
        except Exception:
            return None

    def _judge_with_llm(
        self,
        existing: SkillUnit,
        candidate: SkillUnit,
        local_score: float,
    ) -> dict[str, Any]:
        prompt = (
            "You are maintaining a cumulative micro-skill registry for an ECG data-analysis agent.\n"
            "Decide whether the candidate skill and existing skill describe the same reusable capability.\n"
            "Merge only if they are operationally interchangeable or one clearly generalizes the other.\n"
            "Do not merge if they only share a broad ECG topic, have different inputs, or imply conflicting failure boundaries.\n"
            "Return strict JSON only with keys: decision, reason, merged_name, merged_description, "
            "merged_procedure_summary, merged_applicability, merged_failure_modes, merged_feature_scope.\n"
            "decision must be one of: merge, keep_separate, conflict.\n\n"
            f"local_similarity={local_score:.3f}\n"
            f"existing={json.dumps(existing.to_dict(), ensure_ascii=False)}\n"
            f"candidate={json.dumps(candidate.to_dict(), ensure_ascii=False)}"
        )
        try:
            from utils import reason_client
            from utils.model import qw_plus

            completion = reason_client.chat.completions.create(
                model=qw_plus["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = completion.choices[0].message.content or ""
            return self._parse_llm_json(content)
        except Exception as exc:
            return {
                "decision": "keep_separate",
                "reason": f"LLM semantic merge unavailable: {type(exc).__name__}",
            }

    @staticmethod
    def _parse_llm_json(content: str) -> dict[str, Any]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                return {"decision": "keep_separate", "reason": "LLM did not return JSON"}
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {"decision": "keep_separate", "reason": "LLM JSON parse failed"}

        if parsed.get("decision") not in {"merge", "keep_separate", "conflict"}:
            parsed["decision"] = "keep_separate"
            parsed["reason"] = parsed.get("reason") or "invalid merge decision"
        if not isinstance(parsed.get("merged_failure_modes"), list):
            parsed["merged_failure_modes"] = []
        if not isinstance(parsed.get("merged_feature_scope"), str):
            parsed["merged_feature_scope"] = ""
        return parsed


def _skill_semantic_text(skill: SkillUnit) -> str:
    return " ".join(
        [
            skill.name,
            skill.description,
            skill.feature_scope,
            skill.procedure_summary,
            skill.applicability,
            " ".join(skill.input_requirements),
            " ".join(skill.action_bundle),
            " ".join(skill.validation_basis),
            " ".join(skill.evidence_pattern),
            " ".join(skill.tool_dependencies),
            " ".join(skill.failure_modes),
        ]
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _signature_for_skill(skill: SkillUnit) -> str:
    signature = "|".join(
        [
            skill.agent_name,
            skill.level,
            skill.category,
            _normalize(skill.feature_scope),
            _normalize(skill.name),
            _normalize(skill.procedure_summary),
        ]
    )
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]


def build_family_id(skill: SkillUnit) -> str:
    seed = "|".join(
        [
            skill.agent_name,
            skill.level,
            skill.category,
            _normalize(skill.feature_scope),
            _normalize(skill.name),
        ]
    )
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def build_skill_id(agent_name: str, name: str, procedure_summary: str) -> str:
    seed = f"{agent_name}|{_normalize(name)}|{_normalize(procedure_summary)}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


class SkillExtractor:
    def extract(
        self,
        *,
        sample_id: str,
        question_id: str = "",
        template_id: str = "",
        question: str,
        choices: list[str],
        data_analysis_memory: list[Any],
        final_answer: str | None,
    ) -> list[SkillUnit]:
        texts = collect_texts(data_analysis_memory)
        joined = "\n".join(texts)
        lowered = joined.lower()
        candidates: list[SkillUnit] = []
        now = _utc_now()

        def add_skill(
            *,
            name: str,
            description: str,
            category: str,
            input_requirements: list[str],
            procedure_summary: str,
            evidence_pattern: list[str],
            tool_dependencies: list[str],
            applicability: str,
            failure_modes: list[str],
            confidence: float,
        ) -> None:
            skill_id = build_skill_id("data_analysis_agent", name, procedure_summary)
            family_seed = "|".join(["data_analysis_agent", "micro", category, name])
            family_id = hashlib.sha1(_normalize(family_seed).encode("utf-8")).hexdigest()[:12]
            candidates.append(
                SkillUnit(
                    skill_id=skill_id,
                    agent_name="data_analysis_agent",
                    name=name,
                    description=description,
                    level="micro",
                    category=category,
                    input_requirements=input_requirements,
                    procedure_summary=procedure_summary,
                    evidence_pattern=evidence_pattern,
                    tool_dependencies=tool_dependencies,
                    applicability=applicability,
                    failure_modes=failure_modes,
                    source_sample_ids=[sample_id],
                    source_question_ids=[question_id] if question_id else [],
                    source_task_refs=[
                        {
                            "sample_id": sample_id,
                            "question_id": question_id,
                            "template_id": template_id,
                        }
                    ],
                    family_id=family_id,
                    source_trace_file="agent_skill_sources.jsonl",
                    confidence=confidence,
                    reuse_count=0,
                    success_count=0,
                    failure_count=0,
                    reflection_notes=[],
                    last_reflection="",
                    created_at=now,
                    updated_at=now,
                )
            )

        if "get_lead_signals(" in joined:
            add_skill(
                name="read-single-lead-signal",
                description="Read a single ECG lead waveform using the documented ECG API before computing lead-local evidence.",
                category="measurement",
                input_requirements=["ecg object", "valid lead name"],
                procedure_summary="Call `ecg.get_lead_signals(lead_name)` and limit subsequent analysis to one lead-level claim at a time.",
                evidence_pattern=["get_lead_signals(", "lead signal", "raw signal"],
                tool_dependencies=[],
                applicability="Use when a feature can be measured directly from one lead waveform.",
                failure_modes=["Lead name is unavailable", "Waveform is too noisy for reliable numeric measurement"],
                confidence=0.72,
            )

        if "get_lead_segment(" in joined:
            add_skill(
                name="extract-ecg-segment-window",
                description="Use segment boundaries to isolate a waveform or interval before measurement.",
                category="measurement",
                input_requirements=["ecg object", "valid lead name", "segment name"],
                procedure_summary="Call `ecg.get_lead_segment(lead_name, segment_name)` and derive measurements from one returned `(start, end)` window.",
                evidence_pattern=["get_lead_segment(", "start, end", "segment"],
                tool_dependencies=[],
                applicability="Use when a feature depends on a known ECG segment such as QRS, QT, T wave, or PR.",
                failure_modes=["Segment is not detected", "Requested segment type is unsupported"],
                confidence=0.75,
            )

        if ("duration" in lowered or "ms" in lowered) and ("fs" in lowered or "sampling frequency" in lowered):
            add_skill(
                name="convert-segment-index-to-duration",
                description="Convert ECG sample indices into time duration only when sampling frequency is verified.",
                category="measurement",
                input_requirements=["segment start/end indices", "verified fs"],
                procedure_summary="Compute duration with `(end - start) * 1000 / fs` after confirming that `fs` is available from documented context.",
                evidence_pattern=["1000 / fs", "duration", "ms"],
                tool_dependencies=[],
                applicability="Use for QT, PR, QRS, RR, or similar duration-style questions.",
                failure_modes=["Sampling frequency is missing or undocumented", "Segment boundaries are unreliable"],
                confidence=0.78,
            )

        if ("fs" in lowered and ("not provided" in lowered or "must be obtained" in lowered or "need to inspect" in lowered)):
            add_skill(
                name="refuse-duration-claim-without-fs",
                description="Avoid duration-based conclusions when the task does not provide a verified sampling frequency.",
                category="fallback",
                input_requirements=["duration-like feature request"],
                procedure_summary="Before computing any duration, verify `fs` from documented context; otherwise mark the duration claim unverified instead of hard-coding a value.",
                evidence_pattern=["fs", "not provided", "must be obtained"],
                tool_dependencies=[],
                applicability="Use whenever durations in ms are required but sampling frequency is not documented.",
                failure_modes=["Hidden assumptions about `fs` slip into code", "Undocumented ECG attributes are treated as guaranteed"],
                confidence=0.84,
            )

        if ("ecg_fig_anaysis_tool" in lowered or "visual inspection" in lowered or "noise-sensitive" in lowered):
            add_skill(
                name="route-morphology-checks-to-visual-tool",
                description="Route visually defined or noise-sensitive morphology claims to the ECG figure analysis tool.",
                category="tool-routing",
                input_requirements=["lead waveform", "verified fs for plotting when required", "morphology question"],
                procedure_summary="When the feature is mainly visual or fragile under simple numeric rules, call `ECG_fig_anaysis_tool` and treat the result as supporting evidence.",
                evidence_pattern=["ECG_fig_anaysis_tool", "visual inspection", "noise-sensitive"],
                tool_dependencies=["ECG_fig_anaysis_tool"],
                applicability="Use for scooped ST, subtle contour change, or similar morphology-heavy evidence.",
                failure_modes=["Tool output is treated as direct ground truth", "Too many cycles are passed to the visual tool at once"],
                confidence=0.81,
            )

        if "ecgdeli_measurement_tool" in lowered and (
            "feature_name" in lowered or "wave_name" in lowered or "value" in lowered or "samples" in lowered
        ):
            add_skill(
                name="consume-ecgdeli-measurement-result",
                description="Interpret fine-grained ECGdeli measurement output as task evidence without turning tool selection itself into a learned skill.",
                category="measurement",
                input_requirements=["ECGDeli measurement result JSON", "validated question-to-feature mapping"],
                procedure_summary="Read ECGdeli measurement fields such as `value`, `units`, `feature_name`, `wave_name`, or `samples`, then map only the validated fine-grained result back to the allowed answer choices.",
                evidence_pattern=["ECGDeli_measurement_tool", "feature_name", "wave_name", "value"],
                tool_dependencies=["ECGDeli_measurement_tool"],
                applicability="Use after ECGdeli has already been called and returned a fine-grained interval, amplitude, fiducial-point, or slice result relevant to the task.",
                failure_modes=["ECGDeli measurement status is not ok", "A narrow ECGdeli measurement is over-interpreted beyond the requested feature claim"],
                confidence=0.76,
            )

        choice_hits = [choice for choice in choices if choice.lower() in lowered]
        if choice_hits and ("criterion" in lowered or "criteria" in lowered or "support" in lowered or "if " in lowered):
            add_skill(
                name="map-validated-feature-evidence-to-choice",
                description="Map validated feature evidence back to the allowed answer choices instead of producing free-form diagnoses.",
                category="decision-rule",
                input_requirements=["validated feature evidence", "allowed answer options"],
                procedure_summary="Compare only validated evidence against the provided answer options and select the matching option names explicitly.",
                evidence_pattern=choice_hits[:5],
                tool_dependencies=[],
                applicability="Use for multi-choice ECGQA tasks where answers must come from a fixed option list.",
                failure_modes=["Evidence is weaker than the claimed answer", "The response contains diagnoses outside the provided options"],
                confidence=0.74,
            )

        unique: dict[str, SkillUnit] = {}
        for skill in candidates:
            unique[skill.skill_id] = skill
        return [skill for skill in unique.values() if not self._should_drop_skill(skill)]

    @staticmethod
    def _should_drop_skill(skill: SkillUnit) -> bool:
        joined = " ".join(
            [
                skill.name,
                skill.description,
                skill.procedure_summary,
                skill.applicability,
                " ".join(skill.evidence_pattern),
            ]
        ).lower()
        if "ecgdeli" not in joined:
            return False
        disallowed_phrases = [
            "prefer ecgdeli",
            "route to ecgdeli",
            "when to call ecgdeli",
            "enable ecgdeli",
            "use ecgdeli first",
        ]
        return any(phrase in joined for phrase in disallowed_phrases)


class ReflectiveSkillGenerator:
    _FEATURE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
        ("qrs_duration", ("qrs duration", "qrs", "qrs_complex")),
        ("qt_interval", ("qt interval", "qt", "qtc")),
        ("pr_interval", ("pr interval", "pr", "p-r")),
        ("rr_interval", ("rr interval", "rr", "heart rate")),
        ("t_wave_inversion", ("t wave inversion", "inverted t", "t-wave inversion")),
        ("st_segment_morphology", ("st segment", "st elevation", "st depression", "st morphology")),
        ("p_wave_morphology", ("p wave", "p-wave")),
        ("r_amplitude", ("r amplitude", "r wave amplitude", "high qrs voltage")),
    ]

    def generate(
        self,
        *,
        sample_id: str,
        question_id: str = "",
        template_id: str = "",
        question: str,
        choices: list[str],
        data_analysis_memory: list[Any],
        final_answer: str | None,
        predicted_answer: str | None = None,
        gold_answers: list[str] | None = None,
        mode: str = "success",
    ) -> list[SkillUnit]:
        texts = collect_texts(data_analysis_memory)
        joined = "\n".join(texts)
        lowered = joined.lower()
        question_lower = question.lower()
        now = _utc_now()
        features = self._detect_features(question_lower, lowered)
        if not features:
            features = ["general_ecg_feature"]

        skills: list[SkillUnit] = []
        for feature_scope in features:
            action_bundle = self._infer_action_bundle(question_lower, lowered, feature_scope)
            tool_dependencies = self._infer_tool_dependencies(lowered)
            input_requirements = self._infer_input_requirements(lowered, feature_scope)
            failure_modes = self._infer_failure_modes(lowered, feature_scope)
            evidence_pattern = self._infer_evidence_patterns(choices, lowered, feature_scope)
            procedure_summary = self._build_procedure_summary(feature_scope, action_bundle, mode)
            name = self._build_skill_name(feature_scope, mode)
            description = self._build_description(feature_scope, mode)
            validation_basis = ["final_answer_match"] if mode == "success" else ["gold_answer", "retry_success"]
            status = "active" if mode == "success" else "retry_only"
            skill_source = "success_trace" if mode == "success" else "failure_retry"
            origin_gold = gold_answers or []
            skill_id = build_skill_id("data_analysis_agent", name, procedure_summary)
            family_seed = "|".join(["data_analysis_agent", "micro", "reflective-analysis", feature_scope])
            family_id = hashlib.sha1(_normalize(family_seed).encode("utf-8")).hexdigest()[:12]
            skills.append(
                SkillUnit(
                    skill_id=skill_id,
                    agent_name="data_analysis_agent",
                    name=name,
                    description=description,
                    level="micro",
                    category="reflective-analysis",
                    input_requirements=input_requirements,
                    procedure_summary=procedure_summary,
                    evidence_pattern=evidence_pattern,
                    tool_dependencies=tool_dependencies,
                    applicability=f"Use when the task depends on verifying `{feature_scope}` through a reusable feature-level workflow.",
                    failure_modes=failure_modes,
                    source_sample_ids=[sample_id],
                    source_question_ids=[question_id] if question_id else [],
                    source_task_refs=[
                        {
                            "sample_id": sample_id,
                            "question_id": question_id,
                            "template_id": template_id,
                        }
                    ],
                    confidence=0.78 if mode == "success" else 0.64,
                    reuse_count=0,
                    created_at=now,
                    updated_at=now,
                    family_id=family_id,
                    source_trace_file="agent_skill_sources.jsonl",
                    skill_source=skill_source,
                    feature_scope=feature_scope,
                    action_bundle=action_bundle,
                    validation_basis=validation_basis,
                    status=status,
                    origin_prediction=predicted_answer or "",
                    origin_gold=origin_gold,
                    applies_on_retry_only=mode != "success",
                )
            )
        return [skill for skill in skills if not _is_native_capability_duplicate(skill)]

    def _detect_features(self, question_lower: str, memory_lower: str) -> list[str]:
        matched: list[str] = []
        combined = f"{question_lower}\n{memory_lower}"
        for feature_scope, patterns in self._FEATURE_PATTERNS:
            if any(pattern in combined for pattern in patterns):
                matched.append(feature_scope)
        return matched[:3]

    def _infer_action_bundle(self, question_lower: str, memory_lower: str, feature_scope: str) -> list[str]:
        actions: list[str] = []
        if "get_lead_signals(" in memory_lower:
            actions.append("read target lead waveform before drawing a feature conclusion")
        if "get_lead_segment(" in memory_lower:
            actions.append("slice the narrowest ECG segment associated with the feature")
        if "ecgdeli_measurement_tool" in memory_lower:
            actions.append("consume ECGdeli feature-level measurement output for the target feature")
        if "ecg_fig_anaysis_tool" in memory_lower:
            actions.append("route morphology-sensitive verification to the visual ECG tool")
        if "fs" in memory_lower or any(token in question_lower for token in ["duration", "interval", "qt", "qrs", "pr", "rr"]):
            actions.append("verify sampling frequency and convert sample span into duration only after confirmation")
        actions.append(f"map validated `{feature_scope}` evidence back to the allowed answer choices")
        return _merge_list(actions, [])

    def _infer_tool_dependencies(self, memory_lower: str) -> list[str]:
        tools: list[str] = []
        if "ecgdeli_measurement_tool" in memory_lower:
            tools.append("ECGDeli_measurement_tool")
        if "ecg_fig_anaysis_tool" in memory_lower:
            tools.append("ECG_fig_anaysis_tool")
        return tools

    def _infer_input_requirements(self, memory_lower: str, feature_scope: str) -> list[str]:
        reqs = ["ecg object", f"feature target: {feature_scope}"]
        if "get_lead_signals(" in memory_lower or "get_lead_segment(" in memory_lower:
            reqs.append("valid lead name")
        if "fs" in memory_lower or "interval" in feature_scope or "duration" in feature_scope:
            reqs.append("verified sampling frequency for duration claims")
        return _merge_list(reqs, [])

    def _infer_failure_modes(self, memory_lower: str, feature_scope: str) -> list[str]:
        failures = [
            f"{feature_scope} evidence is mapped to the wrong answer option",
            "task-specific answer text is copied instead of re-deriving evidence",
        ]
        if "fs" in memory_lower:
            failures.append("sampling frequency is assumed instead of verified")
        if "get_lead_segment(" in memory_lower:
            failures.append("segment boundaries are reused without checking they match the feature")
        if "ecg_fig_anaysis_tool" in memory_lower:
            failures.append("visual-tool output is treated as direct truth instead of supporting evidence")
        return _merge_list(failures, [])

    def _infer_evidence_patterns(self, choices: list[str], memory_lower: str, feature_scope: str) -> list[str]:
        evidence = [feature_scope]
        evidence.extend(choice for choice in choices if choice.lower() in memory_lower)
        if "get_lead_segment(" in memory_lower:
            evidence.append("segment-window evidence")
        if "get_lead_signals(" in memory_lower:
            evidence.append("lead-waveform evidence")
        return _merge_list(evidence[:6], [])

    @staticmethod
    def _build_procedure_summary(feature_scope: str, action_bundle: list[str], mode: str) -> str:
        mode_prefix = "Retry-only correction workflow:" if mode != "success" else "Feature workflow:"
        joined_actions = "; ".join(action_bundle[:5])
        return f"{mode_prefix} verify `{feature_scope}` with the smallest valid evidence path, then {joined_actions}."

    @staticmethod
    def _build_skill_name(feature_scope: str, mode: str) -> str:
        prefix = "retry-correct" if mode != "success" else "analyze"
        return f"{prefix}-{feature_scope}"

    @staticmethod
    def _build_description(feature_scope: str, mode: str) -> str:
        if mode == "success":
            return f"Reusable feature-level analysis bundle for `{feature_scope}` extracted from a correct execution trace."
        return f"Temporary corrective feature-level bundle for `{feature_scope}` derived from a failed attempt and gold-answer supervision."


class SkillEvaluator:
    def evaluate(
        self,
        candidates: list[SkillUnit],
        registry: SkillRegistry,
    ) -> tuple[list[SkillUnit], list[tuple[str, str]]]:
        accepted: list[SkillUnit] = []
        rejected: list[tuple[str, str]] = []

        for skill in candidates:
            if not skill.procedure_summary or len(skill.procedure_summary) < 30:
                rejected.append((skill.name, "procedure_summary is too short"))
                continue
            if any(token in skill.procedure_summary.lower() for token in ["sample_id", "this exact question", "this patient"]):
                rejected.append((skill.name, "skill is too task-specific"))
                continue
            if "ecg.fs" in skill.procedure_summary.lower() or ".sampling_rate" in skill.procedure_summary.lower():
                rejected.append((skill.name, "depends on undocumented API access"))
                continue
            if skill.category not in {"measurement", "decision-rule", "tool-routing", "fallback", "reflective-analysis"}:
                rejected.append((skill.name, "unsupported category"))
                continue
            if skill.category == "reflective-analysis":
                if not skill.feature_scope.strip():
                    rejected.append((skill.name, "reflective skill must declare feature_scope"))
                    continue
                if len(skill.action_bundle) < 2:
                    rejected.append((skill.name, "reflective skill must contain at least two actions"))
                    continue
                if skill.status not in {"active", "pending_review", "retry_only"}:
                    rejected.append((skill.name, "invalid reflective skill status"))
                    continue
                if skill.applies_on_retry_only and skill.status == "active":
                    rejected.append((skill.name, "retry-only skill cannot be active"))
                    continue
                if _is_native_capability_duplicate(skill):
                    rejected.append((skill.name, "duplicates native tool/static skill capability"))
                    continue
            stored_skill = registry.upsert_skill(skill)
            accepted.append(stored_skill)

        return accepted, rejected


class SkillRetriever:
    def retrieve(
        self,
        *,
        registry: SkillRegistry,
        question: str,
        pre_analysis_res: str | None,
        top_k: int = 5,
    ) -> list[SkillUnit]:
        return registry.find_relevant_skills(
            question=question,
            pre_analysis_res=pre_analysis_res,
            top_k=top_k,
        )

    def retrieve_with_llm(
        self,
        *,
        skills: list[SkillUnit],
        question: str,
        pre_analysis_res: str | None,
        top_k: int = 5,
    ) -> list[SkillUnit]:
        if not skills or os.environ.get("AGENT_ECG_DISABLE_LLM_SKILL_RETRIEVAL") == "1":
            return []

        catalog = self._build_skill_catalog(skills)
        if not catalog:
            return []

        prompt = (
            "You are selecting reusable micro-skills for an ECG data-analysis agent.\n"
            "Given the current question, pre-analysis, and a skill catalog, return the skill_ids that should be injected.\n"
            "Select only skills that are operationally useful for this task. Do not select skills merely because they share broad ECG words.\n"
            "Prefer skills whose applicability, procedure, or failure modes match the current task.\n"
            "If no skill is useful, return an empty list.\n"
            "Return strict JSON only: {\"skill_ids\": [\"...\"], \"reason\": \"...\"}.\n\n"
            f"top_k={top_k}\n"
            f"question={question}\n"
            f"pre_analysis={pre_analysis_res or ''}\n"
            f"skill_catalog={json.dumps(catalog, ensure_ascii=False)}"
        )
        try:
            from utils import reason_client
            from utils.model import qw_plus

            completion = reason_client.chat.completions.create(
                model=qw_plus["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = completion.choices[0].message.content or ""
            selected_ids = self._parse_selected_skill_ids(content)
        except Exception:
            return []

        if not selected_ids:
            return []
        skill_by_id = {skill.skill_id: skill for skill in skills}
        selected = []
        for skill_id in selected_ids:
            skill = skill_by_id.get(skill_id)
            if skill is not None and skill not in selected:
                selected.append(skill)
            if len(selected) >= top_k:
                break
        return selected

    @staticmethod
    def _build_skill_catalog(skills: list[SkillUnit], max_items: int = 80) -> list[dict[str, Any]]:
        sorted_skills = sorted(
            skills,
            key=lambda skill: (skill.reuse_count, skill.confidence, skill.updated_at),
            reverse=True,
        )
        catalog = []
        for skill in sorted_skills[:max_items]:
            catalog.append(
                {
                    "skill_id": skill.skill_id,
                    "name": skill.name,
                    "category": skill.category,
                    "description": skill.description,
                    "procedure_summary": skill.procedure_summary,
                    "applicability": skill.applicability,
                    "failure_modes": skill.failure_modes[:3],
                    "tool_dependencies": skill.tool_dependencies,
                    "confidence": round(skill.confidence, 3),
                    "reuse_count": skill.reuse_count,
                    "conflict_with": skill.conflict_with,
                }
            )
        return catalog

    @staticmethod
    def _parse_selected_skill_ids(content: str) -> list[str]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                return []
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []
        skill_ids = parsed.get("skill_ids", [])
        if not isinstance(skill_ids, list):
            return []
        return [str(skill_id) for skill_id in skill_ids if isinstance(skill_id, str)]


class SkillComposer:
    def compose(self, skills: list[SkillUnit]) -> str:
        if not skills:
            return "Relevant micro-skills:\n- None retrieved from prior tasks."

        lines = ["Relevant micro-skills:"]
        for skill in skills:
            lines.append(f"- {skill.name} [{skill.category}]")

        lines.append("How to apply:")
        for skill in skills:
            lines.append(f"- {skill.procedure_summary}")
            if skill.action_bundle:
                lines.append(f"- {skill.name} actions: {'; '.join(skill.action_bundle[:4])}")

        lines.append("Known limits:")
        for skill in skills:
            limit = "; ".join(skill.failure_modes) if skill.failure_modes else "No explicit limits recorded."
            lines.append(f"- {skill.name}: {limit}")

        conflicted = [skill for skill in skills if skill.conflict_with]
        if conflicted:
            lines.append("Known conflicts requiring review:")
            for skill in conflicted:
                conflicts = ", ".join(skill.conflict_with)
                lines.append(f"- {skill.name}: conflicts with {conflicts}; verify prerequisites before applying.")

        lines.append("When to route to tool:")
        routed = [skill for skill in skills if "ECG_fig_anaysis_tool" in skill.tool_dependencies]
        if routed:
            for skill in routed:
                lines.append(f"- {skill.name}: {skill.applicability}")
        else:
            lines.append("- No prior tool-routing micro-skill retrieved for this task.")

        return "\n".join(lines)

    def compose_pre_analysis(self, skills: list[SkillUnit]) -> str:
        if not skills:
            return "Local reusable skill plan:\n- None retrieved from prior tasks."

        lines = ["Local reusable skill plan:"]
        lines.append("Candidate feature workflows:")
        for skill in skills:
            feature_scope = skill.feature_scope or skill.name
            lines.append(f"- {feature_scope}: {skill.description}")

        lines.append("Preferred local tool paths:")
        for skill in skills:
            if skill.tool_dependencies:
                lines.append(f"- {skill.name}: prefer {' -> '.join(skill.tool_dependencies)}")
            else:
                lines.append(f"- {skill.name}: use documented ECG APIs before considering remote search")

        lines.append("Known local limits:")
        for skill in skills:
            limit = "; ".join(skill.failure_modes) if skill.failure_modes else "No explicit limits recorded."
            lines.append(f"- {skill.name}: {limit}")

        conflicted = [skill for skill in skills if skill.conflict_with]
        if conflicted:
            lines.append("Conflicts that may require remote medical disambiguation:")
            for skill in conflicted:
                conflicts = ", ".join(skill.conflict_with)
                lines.append(f"- {skill.name}: conflicts with {conflicts}")
        else:
            lines.append("Conflicts that may require remote medical disambiguation:")
            lines.append("- No local skill conflicts detected.")

        lines.append("When remote search is still justified:")
        lines.append("- Only for ECG definitions, thresholds, diagnostic criteria, lead-specific interpretation rules, or conflict resolution.")
        lines.append("- Do not use remote search to replace local tool usage or generate alternative code recipes.")
        return "\n".join(lines)


class PendingSkillGenerator:
    def compose_retry_context(self, skills: list[SkillUnit], failed_answer: str | None) -> str:
        if not skills:
            return self.generate(
                question="",
                choices=[],
                failed_answer=failed_answer,
                failure_reflection={"root_cause": "retry requested without a generated corrective skill"},
                data_analysis_memory=[],
            )

        lines = [
            "Pending micro-skills from failed attempt:",
            "- These skills are retry-only. Use them to re-validate evidence, not as facts about the current ECG.",
            f"- Previous answer to avoid blindly repeating: {failed_answer or 'None'}",
            "How to apply:",
        ]
        for skill in skills:
            lines.append(f"- {skill.name} [{skill.feature_scope or skill.category}]")
            lines.append(f"- {skill.procedure_summary}")
            if skill.action_bundle:
                lines.append(f"- Actions: {'; '.join(skill.action_bundle[:5])}")
        lines.append("Retry constraints:")
        lines.append("- Do not copy the gold answer into the final response.")
        lines.append("- Re-derive the answer from feature-level evidence and allowed choices only.")
        return "\n".join(lines)

    def generate(
        self,
        *,
        question: str,
        choices: list[str],
        failed_answer: str | None,
        failure_reflection: dict[str, str],
        data_analysis_memory: list[Any],
        variable_summaries: list[dict[str, Any]] | None = None,
        vlm_slice_audit: dict[str, Any] | None = None,
    ) -> str:
        texts = collect_texts(data_analysis_memory)
        joined = "\n".join(texts).lower()
        audit_verdict = (vlm_slice_audit or {}).get("verdict", "uncertain")
        lines = [
            "Pending micro-skills from failed attempt:",
            "- retry-with-narrower-evidence [fallback]",
            "How to apply:",
            "- Treat the previous attempt as untrusted. Re-check the smallest feature-level evidence needed for this question before selecting an option.",
            "- Do not reuse the previous answer directly; select only from the provided choices after re-validating evidence.",
            f"- Previous answer to avoid blindly repeating: {failed_answer or 'None'}",
            "Known limits:",
            f"- Failure signal: {failure_reflection.get('root_cause', 'prediction did not match evaluator')}",
        ]

        if variable_summaries:
            checked_names = ", ".join(
                str(item.get("name", "unknown")) for item in variable_summaries[:8]
            )
            lines.append(f"- Available checked intermediate variables: {checked_names}")

        if audit_verdict in {"valid", "uncertain"}:
            audit_reason = (vlm_slice_audit or {}).get("reason", "No high-confidence slicing error was found.")
            morphology_assessment = (vlm_slice_audit or {}).get("morphology_assessment", "")
            supports_target_feature = (vlm_slice_audit or {}).get("supports_target_feature")
            lines.extend(
                [
                    "- vlm-slice-audit-awareness [fallback]",
                    f"- VLM slice audit verdict was `{audit_verdict}`: {audit_reason}",
                    (
                        f"- Morphology assessment: {morphology_assessment}"
                        if morphology_assessment
                        else "- Morphology assessment was inconclusive; prioritize direct feature support checks."
                    ),
                    (
                        "- The checked slice did not clearly support the target feature; re-check morphology and feature selection before trusting downstream mapping."
                        if supports_target_feature is False
                        else "- Do not assume the previous slice boundaries were the main failure source unless new evidence proves it; prioritize threshold direction, answer-option mapping, and feature selection checks."
                    ),
                ]
            )

        if "fs" in joined or any(token in question.lower() for token in ["duration", "interval", "qt", "qrs", "pr"]):
            lines.extend(
                [
                    "- verify-duration-prerequisites [measurement]",
                    "- Before any duration or interval conclusion, verify sampling frequency and segment boundaries from documented context. If `fs` is missing, do not hard-code it.",
                ]
            )

        if "get_lead_segment(" in joined or "get_lead_signals(" in joined:
            lines.extend(
                [
                    "- audit-measurement-to-choice-mapping [decision-rule]",
                    "- Check whether the measured evidence actually supports the selected option, especially threshold direction such as above/below/within normal range.",
                ]
            )

        if "ecg_fig_anaysis_tool" in joined or any(token in question.lower() for token in ["morphology", "st", "t wave", "p wave", "noise"]):
            lines.extend(
                [
                    "When to route to tool:",
                    "- If the feature is visual, morphology-heavy, or noise-sensitive, route the narrowest lead/window to `ECG_fig_anaysis_tool` and treat output as supporting evidence, not direct truth.",
                ]
            )

        allowed = ", ".join(str(choice) for choice in choices)
        lines.append(f"Allowed choices for retry: {allowed}")
        return "\n".join(lines)


class ReflectionGenerator:
    def generate(
        self,
        *,
        question: str,
        predicted_answer: str | None,
        gold_answers: list[str],
        is_correct: bool,
        data_analysis_memory: list[Any],
        used_skill_names: list[str],
        learned_skill_names: list[str],
        variable_summaries: list[dict[str, Any]] | None = None,
        vlm_slice_audit: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        texts = collect_texts(data_analysis_memory)
        joined = "\n".join(texts).lower()
        predicted = predicted_answer or "None"
        gold = ", ".join(gold_answers) if gold_answers else "None"
        audit_verdict = (vlm_slice_audit or {}).get("verdict", "")
        audited_variables = ", ".join(
            str(item.get("name", "unknown")) for item in (variable_summaries or [])[:8]
        ) or "none"
        audit_reason = str((vlm_slice_audit or {}).get("reason", "")).strip()
        morphology_assessment = str((vlm_slice_audit or {}).get("morphology_assessment", "")).strip()
        supports_target_feature = (vlm_slice_audit or {}).get("supports_target_feature")

        if is_correct:
            summary = (
                f"Prediction matched the gold answer. Reusable skills that contributed to this task "
                f"should be reinforced and reused on similar questions."
            )
            root_cause = "Correct outcome with task-grounded evidence."
            action = (
                "Keep the retrieved micro-skills, prefer the same measurement or routing pattern on similar tasks, "
                "and raise confidence for the skills involved."
            )
        else:
            likely_issue = self._infer_failure_root_cause(
                joined=joined,
                audit_verdict=audit_verdict,
                audit_reason=audit_reason,
                morphology_assessment=morphology_assessment,
                supports_target_feature=supports_target_feature,
            )

            summary = (
                f"Prediction did not match the gold answer. The current task should produce corrective feedback so the reused skills "
                f"do not become over-trusted."
            )
            root_cause = likely_issue
            action = (
                "Lower confidence for reused skills on this task, keep the failure mode explicit, and prefer narrower validation "
                "before selecting an answer on similar questions."
            )

        used = ", ".join(used_skill_names) if used_skill_names else "none"
        learned = ", ".join(learned_skill_names) if learned_skill_names else "none"

        return {
            "summary": summary,
            "comparison": f"predicted={predicted}; gold={gold}",
            "root_cause": root_cause,
            "used_skills": used,
            "learned_skills": learned,
            "next_action": action,
            "question": question,
            "audited_variables": audited_variables,
            "vlm_slice_audit": json.dumps(vlm_slice_audit or {}, ensure_ascii=False),
        }

    @staticmethod
    def _infer_failure_root_cause(
        *,
        joined: str,
        audit_verdict: str,
        audit_reason: str,
        morphology_assessment: str,
        supports_target_feature: bool | None,
    ) -> str:
        if audit_verdict == "segmentation_error":
            return (
                "VLM morphology audit indicates a segment-boundary error that makes the current slice unsuitable for the target feature. "
                + (audit_reason or "The current evidence path is unsalvageable without a different segmentation result.")
            )

        morphology_clause = ""
        if morphology_assessment:
            morphology_clause = f" VLM morphology assessment: {morphology_assessment}."
        elif audit_reason:
            morphology_clause = f" VLM audit signal: {audit_reason}."

        if supports_target_feature is False:
            return (
                "VLM morphology audit suggests the checked slice does not support the target feature, so the failure is more likely due to "
                "feature selection, morphology interpretation, or answer-option mapping rather than a pure slicing failure."
                + morphology_clause
            )
        if supports_target_feature is True:
            return (
                "VLM morphology audit suggests the checked slice is broadly compatible with the target feature, so the failure is more likely due to "
                "threshold direction, evidence weighting, or answer-option mapping after measurement."
                + morphology_clause
            )
        if "fs" in joined and ("not provided" in joined or "must be obtained" in joined):
            return (
                "A duration-style conclusion may have depended on missing or weakly verified sampling-frequency evidence."
                + morphology_clause
            )
        if "ecg_fig_anaysis_tool" in joined:
            return (
                "The task may have leaned too heavily on tool-routed morphology evidence or interpreted tool evidence too strongly."
                + morphology_clause
            )
        if "get_lead_segment(" in joined or "get_lead_signals(" in joined:
            return (
                "The feature extraction path may have measured the wrong evidence or mapped valid evidence to the wrong option."
                + morphology_clause
            )
        if audit_verdict in {"valid", "uncertain"}:
            return (
                "The agent selected an answer that did not exactly match the gold answer, and VLM did not confirm a segment-boundary failure."
                + morphology_clause
            )
        return "The agent selected an answer that did not exactly match the gold answer."
