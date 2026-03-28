import copy
import csv
import json
import random
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from data.EcgSignals import EcgSignals


DEFAULT_DATASET_DIR = "dataset/ecgqa_ptbxl/paraphrased/train"
DEFAULT_SNAPSHOT_ROOT = "tmp/web_snapshots"
DEFAULT_UPLOAD_ROOT = "tmp/web_uploads"


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    tmp_path.replace(target)


def load_json_file(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_text_file(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_version_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class ECGSampleStore:
    """
    仅供 snapshot 构建阶段使用；运行时 Web 不再直接读取数据集目录。
    """

    def __init__(self, dataset_dir: str = DEFAULT_DATASET_DIR, rng: random.Random | None = None):
        self.dataset_dir = Path(dataset_dir)
        self.rng = rng or random.Random()
        self._records_by_sample_id: dict[str, dict[str, Any]] = {}
        self._build_index()

    def _build_index(self) -> None:
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        for json_path in sorted(self.dataset_dir.glob("*.json")):
            rows = load_json_file(json_path)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict) or "sample_id" not in row or "ecg_path" not in row:
                    continue
                self._records_by_sample_id[str(row.get("sample_id"))] = copy.deepcopy(row)

    def get_sample_by_id(self, sample_id: str | int) -> dict[str, Any]:
        record = self._records_by_sample_id.get(str(sample_id))
        if record is None:
            raise LookupError(f"Sample not found: {sample_id}")
        sample = copy.deepcopy(record)
        sample["ecg_datas"] = self._load_ecg_data(sample.get("ecg_path", []))
        return sample

    @staticmethod
    def _load_ecg_data(ecg_paths: list[str]) -> list["EcgSignals"]:
        try:
            import wfdb
        except ImportError as exc:
            raise RuntimeError("wfdb is required to load ECG samples for snapshot publishing.") from exc

        from data.EcgSignals import EcgSignals

        ecg_datas: list[EcgSignals] = []
        for ecg_path in ecg_paths:
            signals, fields = wfdb.rdsamp(ecg_path)
            ecg_datas.append(EcgSignals(signals=signals, fields=fields))
        return ecg_datas


def render_ecg_image(ecg: Any, target_path: str | Path, title: str = "ECG") -> None:
    signals = np.asarray(getattr(ecg, "signals"))
    leads = list(getattr(ecg, "leads", []))
    fs = int(getattr(ecg, "fs", 500) or 500)
    if signals.ndim != 2:
        raise ValueError("ECG signals must be a 2D array with shape (n_samples, n_leads).")

    n_samples, n_leads = signals.shape
    if not leads or len(leads) != n_leads:
        leads = [f"Lead {idx + 1}" for idx in range(n_leads)]

    time_axis = np.arange(n_samples) / fs
    fig, axes = plt.subplots(n_leads, 1, figsize=(14, max(10, n_leads * 1.2)), sharex=True)
    if n_leads == 1:
        axes = [axes]

    for lead_idx, ax in enumerate(axes):
        ax.plot(time_axis, signals[:, lead_idx], color="#0f4c81", linewidth=0.8)
        ax.set_ylabel(leads[lead_idx], rotation=0, labelpad=28, fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.margins(x=0)

    axes[0].set_title(f"{title} | fs={fs}Hz | samples={n_samples}", fontsize=12)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(target_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_uploaded_ecg_csv(path: str | Path, fs: int = 500) -> "EcgSignals":
    from data.EcgSignals import EcgSignals

    csv_path = Path(path)
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.reader(handle) if row]

    if not rows:
        raise ValueError("Uploaded CSV is empty.")

    header = rows[0]
    has_header = False
    try:
        [float(cell) for cell in header]
    except ValueError:
        has_header = True

    data_rows = rows[1:] if has_header else rows
    if not data_rows:
        raise ValueError("Uploaded CSV has no signal rows.")

    signals = np.asarray([[float(cell) for cell in row] for row in data_rows], dtype=float)
    if signals.ndim == 1:
        signals = signals[:, np.newaxis]
    if signals.ndim != 2:
        raise ValueError("Uploaded CSV must decode to a 2D signal matrix.")
    if signals.shape[0] < 2 or signals.shape[1] < 1:
        raise ValueError("Uploaded CSV must contain at least two samples and one lead.")

    if has_header:
        leads = [str(cell).strip() or f"Lead {idx + 1}" for idx, cell in enumerate(header[: signals.shape[1]])]
    else:
        leads = [f"Lead {idx + 1}" for idx in range(signals.shape[1])]

    fields = {"sig_name": leads, "fs": int(fs)}
    return EcgSignals(signals=signals, fields=fields)


def normalize_text(value: str) -> str:
    cleaned = " ".join(value.replace("\n", " ").split())
    return cleaned[:280]


def collect_text_fragments(node: Any) -> list[str]:
    fragments: list[str] = []
    if isinstance(node, str):
        fragments.append(node)
    elif isinstance(node, dict):
        for value in node.values():
            fragments.extend(collect_text_fragments(value))
    elif isinstance(node, (list, tuple)):
        for item in node:
            fragments.extend(collect_text_fragments(item))
    return fragments


def extract_memory_conclusion(agent_memories: dict[str, Any]) -> str:
    data_memory = agent_memories.get("data_analysis_agent") or []
    texts = collect_text_fragments(data_memory)
    for text in reversed(texts):
        normalized = normalize_text(text)
        if not normalized:
            continue
        lowered = normalized.lower()
        if any(token in lowered for token in ["final answer", "最终", "therefore", "combined with", "criterion", "criteria"]):
            return normalized
    for text in reversed(texts):
        normalized = normalize_text(text)
        if normalized:
            return normalized
    return ""


def _flatten_step_texts(node: Any, *, prefixes: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str]]:
    rows: list[tuple[tuple[str, ...], str]] = []
    if isinstance(node, str):
        rows.append((prefixes, node))
        return rows
    if isinstance(node, dict):
        for key, value in node.items():
            rows.extend(_flatten_step_texts(value, prefixes=(*prefixes, str(key))))
        return rows
    if isinstance(node, (list, tuple)):
        for index, item in enumerate(node):
            rows.extend(_flatten_step_texts(item, prefixes=(*prefixes, str(index))))
    return rows


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _looks_like_prompt_or_planning_text(text: str) -> bool:
    if len(text) > 500:
        return True
    blocked_tokens = (
        "you are an expert assistant",
        "you are a world expert",
        "facts survey",
        "plan of action",
        "thought:",
        "code:",
        "observation:",
        "execution requirements",
        "available local analysis tool",
        "this is the question you need to answer",
        "please choose one or more answers from the following options",
        "relevant micro-skills",
        "ecg tool routing verdict",
        "tool_routing_summary",
        "local pre-analysis context",
        "open_questions_for_remote_search",
        "candidate_answers_count",
        "task:",
    )
    return _contains_any(text, blocked_tokens)


def _is_conclusion_like(text: str) -> bool:
    positive_tokens = (
        "final answer",
        "最终答案",
        "therefore",
        "in conclusion",
        "the answer is",
        "predicted answer",
        "诊断",
        "结论",
        "supports",
        "consistent with",
        "indicates",
        "suggests",
        "shows",
        "reveals",
        "measured",
        "measurement",
        "ms",
    )
    return _contains_any(text, positive_tokens)


def _is_observation_like(path_tokens: tuple[str, ...], text: str) -> bool:
    path_text = " ".join(path_tokens).lower()
    if "observations" in path_text or "action_output" in path_text:
        return True
    return _contains_any(
        text,
        (
            "observation",
            "measured",
            "measurement",
            "ms",
            "amplitude",
            "interval",
            "duration",
            "detected",
            "lead",
            "qrs",
            "qt",
            "pr",
            "rr",
            "wave",
        ),
    )


def _extract_memory_evidence_candidates(agent_memories: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    data_memory = agent_memories.get("data_analysis_agent") or []
    for step_index, step in enumerate(data_memory):
        flattened = _flatten_step_texts(step)
        for path_tokens, raw_text in flattened:
            normalized = normalize_text(raw_text)
            if not normalized:
                continue
            if _looks_like_prompt_or_planning_text(normalized):
                continue

            path_text = " ".join(path_tokens).lower()
            score = 0
            if "action_output" in path_text:
                score += 6
            if "model_output" in path_text or "model_output_message" in path_text:
                score += 4
            if "observations" in path_text:
                score += 3
            if _is_conclusion_like(normalized):
                score += 4
            if _is_observation_like(path_tokens, normalized):
                score += 2
            if score <= 0:
                continue

            candidates.append(
                {
                    "step_index": step_index,
                    "path": path_text,
                    "text": normalized,
                    "score": score,
                }
            )
    return candidates


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: (-item["score"], -item["step_index"], item["text"])):
        key = candidate["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _missing_evidence_payload() -> dict[str, Any]:
    return {
        "summary": "未生成可信依据摘要。",
        "bullets": [],
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        loaded = json.loads(stripped)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            return None
        try:
            loaded = json.loads(match.group(0))
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            return None


def _prepare_memory_summary_inputs(agent_memories: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = _dedupe_candidates(_extract_memory_evidence_candidates(agent_memories))
    if not candidates:
        return []

    selected = sorted(candidates[:6], key=lambda item: (item["step_index"], -item["score"]))
    prepared: list[dict[str, Any]] = []
    for item in selected:
        source = "other"
        if "action_output" in item["path"]:
            source = "action_output"
        elif "observations" in item["path"]:
            source = "observations"
        elif "model_output" in item["path"] or "model_output_message" in item["path"]:
            source = "model_output"
        prepared.append(
            {
                "step": int(item["step_index"]),
                "source": source,
                "text": item["text"][:360],
            }
        )
    return prepared


def _call_evidence_summary_model(
    *,
    question: str,
    predicted_answer: str,
    snippets: list[dict[str, Any]],
) -> dict[str, Any]:
    from utils import reason_client

    snippet_lines = []
    for snippet in snippets:
        snippet_lines.append(
            f"- step={snippet['step']} source={snippet['source']}: {snippet['text']}"
        )

    prompt = "\n".join(
        [
            "请根据以下 ECG 推理中间结果，为前端生成依据摘要。",
            "只允许依据给定中间结果，不要补充外部医学知识，不要复述系统提示，不要解释工具选择。",
            "输出必须是一个 JSON 对象，且只能包含这三个键：summary, bullets, confidence。",
            '其中 confidence 只能是 "high"、"medium"、"low"。',
            "summary 应为 1 句高信号依据说明；bullets 为 0 到 3 条简短依据；不要包含 Markdown 代码块。",
            f"问题：{question}",
            f"预测答案：{predicted_answer}",
            "中间结果：",
            *snippet_lines,
        ]
    )
    completion = reason_client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "system",
                "content": "You generate concise, evidence-only JSON summaries from ECG reasoning traces.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    content = completion.choices[0].message.content
    payload = _extract_json_object(content or "")
    if payload is None:
        raise ValueError("Model did not return a valid JSON object.")
    return payload


def _validate_evidence_summary_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    summary = normalize_text(str(payload.get("summary", "")))
    confidence = str(payload.get("confidence", "")).strip().lower()
    bullets_raw = payload.get("bullets") or []
    if confidence not in {"high", "medium", "low"}:
        return None
    if not summary or _looks_like_prompt_or_planning_text(summary):
        return None
    if not (_is_conclusion_like(summary) or _is_observation_like(("summary",), summary)):
        return None
    bullets: list[str] = []
    if isinstance(bullets_raw, list):
        for item in bullets_raw[:3]:
            normalized = normalize_text(str(item))
            if not normalized or normalized == summary or _looks_like_prompt_or_planning_text(normalized):
                continue
            bullets.append(normalized)
    return {
        "summary": summary,
        "bullets": bullets,
        "confidence": confidence,
    }


def build_evidence_payload(
    result_entry: dict[str, Any],
    *,
    agent_memories: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str | None]:
    question = normalize_text(str(result_entry.get("question", "")))
    predicted_answer = normalize_text(str(result_entry.get("final_res", "")))
    if agent_memories is None:
        return _missing_evidence_payload(), "memory_missing"
    if not agent_memories:
        return _missing_evidence_payload(), "memory_unmatched_or_empty"

    snippets = _prepare_memory_summary_inputs(agent_memories)
    if not snippets:
        return _missing_evidence_payload(), "memory_no_extractable_evidence"
    try:
        model_payload = _call_evidence_summary_model(
            question=question,
            predicted_answer=predicted_answer,
            snippets=snippets,
        )
    except Exception:
        return _missing_evidence_payload(), "summary_generation_failed"
    validated = _validate_evidence_summary_payload(model_payload)
    if validated is None:
        return _missing_evidence_payload(), "summary_generation_invalid"
    return {
        "summary": validated["summary"],
        "bullets": validated["bullets"],
    }, None


class SnapshotPublisher:
    def __init__(
        self,
        *,
        dataset_dir: str = DEFAULT_DATASET_DIR,
        snapshot_root: str = DEFAULT_SNAPSHOT_ROOT,
    ):
        self.dataset_dir = dataset_dir
        self.snapshot_root = Path(snapshot_root)
        self.snapshot_base_dir = self.snapshot_root / "snapshots"
        self.manifest_path = self.snapshot_root / "manifest.json"
        self.sample_store = ECGSampleStore(dataset_dir=dataset_dir)

    def publish(
        self,
        *,
        result_file: str,
        agent_mem_file: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        version_id = version or utc_version_id()
        created_at = utc_now_iso()
        version_dir = self.snapshot_base_dir / version_id
        images_dir = version_dir / "images"
        report_path = version_dir / "build_report.json"
        snapshot_path = version_dir / "snapshot.json"

        if version_dir.exists():
            raise FileExistsError(f"Snapshot version already exists: {version_id}")

        version_dir.mkdir(parents=True, exist_ok=False)
        images_dir.mkdir(parents=True, exist_ok=False)

        rows = load_json_file(result_file)
        if not isinstance(rows, list):
            raise ValueError(f"Result file must contain a JSON list: {result_file}")

        all_memories: dict[str, Any] = {}
        if agent_mem_file and Path(agent_mem_file).exists():
            loaded = load_json_file(agent_mem_file)
            if isinstance(loaded, dict):
                all_memories = loaded

        samples: list[dict[str, Any]] = []
        question_type_counts: dict[str, int] = {}
        report: dict[str, Any] = {
            "version": version_id,
            "created_at": created_at,
            "dataset_dir": self.dataset_dir,
            "result_file": str(result_file),
            "agent_mem_file": str(agent_mem_file) if agent_mem_file else None,
            "input_rows": len(rows),
            "published_samples": 0,
            "skipped_samples": [],
            "degraded_samples": [],
        }

        for row in rows:
            if not isinstance(row, dict):
                report["skipped_samples"].append({"sample_id": None, "reason": "invalid_result_row"})
                continue
            sample_id = str(row.get("sample_id", ""))
            try:
                sample = self.sample_store.get_sample_by_id(sample_id)
            except Exception as exc:
                report["skipped_samples"].append({"sample_id": sample_id, "reason": f"dataset_lookup_failed: {exc}"})
                continue

            question_type = sample.get("question_type")
            question = sample.get("question")
            choices = list(sample.get("template_answer") or row.get("template_answer") or [])
            gold_answers = list(row.get("answer") or sample.get("answer") or [])
            predicted_answer = row.get("final_res")

            if not sample_id or not question_type or not question or not choices:
                report["skipped_samples"].append({"sample_id": sample_id or None, "reason": "missing_required_fields"})
                continue

            ecg_refs: list[dict[str, str]] = []
            image_failed = False
            for idx, ecg in enumerate(sample.get("ecg_datas", [])):
                rel_path = Path("images") / f"{sample_id}_{idx}.png"
                abs_path = version_dir / rel_path
                try:
                    render_ecg_image(ecg, abs_path, title=f"ECG {idx + 1}")
                except Exception as exc:
                    report["skipped_samples"].append({"sample_id": sample_id, "reason": f"image_render_failed: {exc}"})
                    image_failed = True
                    break
                ecg_refs.append({"title": f"ECG {idx + 1}", "path": rel_path.as_posix()})

            if image_failed or not ecg_refs:
                continue

            agent_memories = all_memories.get(sample_id) if all_memories else None
            evidence, degradation_reason = build_evidence_payload(row, agent_memories=agent_memories)
            if degradation_reason:
                report["degraded_samples"].append({"sample_id": sample_id, "reason": degradation_reason})

            is_correct = row.get("is_correct")
            if is_correct is None:
                normalized_prediction = str(predicted_answer).strip().lower()
                is_correct = normalized_prediction in {
                    str(answer).strip().lower() for answer in gold_answers if isinstance(answer, str)
                }

            snapshot_row = {
                "sample_id": sample_id,
                "question_id": str(sample.get("question_id", row.get("question_id", ""))),
                "template_id": str(sample.get("template_id", row.get("template_id", ""))),
                "question_type": question_type,
                "question": question,
                "choices": choices,
                "predicted_answer": predicted_answer,
                "gold_answers": gold_answers,
                "is_correct": bool(is_correct),
                "evidence": evidence,
                "ecg_refs": ecg_refs,
            }
            samples.append(snapshot_row)
            question_type_counts[question_type] = question_type_counts.get(question_type, 0) + 1

        snapshot_payload = {
            "version": version_id,
            "created_at": created_at,
            "dataset_dir": self.dataset_dir,
            "question_types": [
                {"name": name, "count": question_type_counts[name]}
                for name in sorted(question_type_counts)
            ],
            "sample_count": len(samples),
            "samples": samples,
        }
        report["published_samples"] = len(samples)

        if not samples:
            atomic_write_json(report_path, report)
            raise RuntimeError(f"Snapshot publish produced no valid samples: {version_id}")

        atomic_write_json(snapshot_path, snapshot_payload)
        atomic_write_json(report_path, report)
        atomic_write_json(self.manifest_path, {"current_version": version_id})
        return {
            "version": version_id,
            "snapshot_dir": str(version_dir),
            "snapshot_file": str(snapshot_path),
            "manifest_file": str(self.manifest_path),
            "build_report_file": str(report_path),
            "sample_count": len(samples),
        }


class SnapshotRuntimeStore:
    def __init__(self, snapshot_root: str = DEFAULT_SNAPSHOT_ROOT, rng: random.Random | None = None):
        self.snapshot_root = Path(snapshot_root)
        self.manifest_path = self.snapshot_root / "manifest.json"
        self.rng = rng or random.Random()
        self.current_version = ""
        self.snapshot_dir: Path | None = None
        self.snapshot_payload: dict[str, Any] | None = None
        self._samples: list[dict[str, Any]] = []
        self._samples_by_type: dict[str, list[dict[str, Any]]] = {}
        self.reload()

    def reload(self) -> None:
        if not self.manifest_path.exists():
            raise FileNotFoundError("尚未发布离线数据：manifest.json 不存在。")

        manifest = load_json_file(self.manifest_path)
        if not isinstance(manifest, dict) or not manifest.get("current_version"):
            raise ValueError("离线数据 manifest 无效：缺少 current_version。")

        version = str(manifest["current_version"])
        snapshot_dir = self.snapshot_root / "snapshots" / version
        snapshot_file = snapshot_dir / "snapshot.json"
        if not snapshot_dir.exists() or not snapshot_file.exists():
            raise FileNotFoundError(f"当前离线版本损坏：{version}")

        payload = load_json_file(snapshot_file)
        if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
            raise ValueError(f"snapshot.json 无效：{snapshot_file}")

        self.current_version = version
        self.snapshot_dir = snapshot_dir
        self.snapshot_payload = payload
        self._samples = payload["samples"]
        grouped: dict[str, list[dict[str, Any]]] = {}

        for row in self._samples:
            question_type = str(row.get("question_type", ""))
            grouped.setdefault(question_type, []).append(row)

        self._samples_by_type = grouped

    def get_options(self) -> dict[str, Any]:
        if self.snapshot_payload is None:
            raise RuntimeError("Snapshot not loaded.")
        return {
            "version": self.snapshot_payload.get("version", self.current_version),
            "created_at": self.snapshot_payload.get("created_at", ""),
            "sample_count": self.snapshot_payload.get("sample_count", len(self._samples)),
            "question_types": self.snapshot_payload.get("question_types", []),
        }

    def get_random_sample(self, question_type: str | None = None) -> dict[str, Any]:
        if question_type:
            pool = self._samples_by_type.get(question_type, [])
        else:
            pool = self._samples
        if not pool:
            raise LookupError("当前离线版本中没有匹配的样本。")
        return copy.deepcopy(self.rng.choice(pool))

    def build_frontend_payload(self, sample_row: dict[str, Any]) -> dict[str, Any]:
        ecg_images = []
        for idx, ref in enumerate(sample_row.get("ecg_refs", [])):
            rel_path = ref.get("path", "")
            token = f"{self.current_version}:{rel_path}"
            ecg_images.append(
                {
                    "index": idx,
                    "title": ref.get("title", f"ECG {idx + 1}"),
                    "image_url": f"/api/ecg/image/{token}",
                }
            )

        return {
            "sample_id": str(sample_row.get("sample_id", "")),
            "question_id": str(sample_row.get("question_id", "")),
            "template_id": str(sample_row.get("template_id", "")),
            "question_type": sample_row.get("question_type"),
            "question": sample_row.get("question"),
            "choices": list(sample_row.get("choices") or []),
            "predicted_answer": sample_row.get("predicted_answer"),
            "gold_answers": list(sample_row.get("gold_answers") or []),
            "is_correct": bool(sample_row.get("is_correct")),
            "ecg_images": ecg_images,
            "evidence": sample_row.get("evidence") or {},
            "snapshot_version": self.current_version,
            "snapshot_created_at": self.snapshot_payload.get("created_at", "") if self.snapshot_payload else "",
        }

    def resolve_image(self, token: str) -> Path:
        version, sep, rel_path = token.partition(":")
        if not sep or not version or not rel_path:
            raise KeyError(token)
        path = self.snapshot_root / "snapshots" / version / rel_path
        if not path.exists():
            raise KeyError(token)
        return path
