import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ECG_SIGNALS_MODULE = ModuleType("data.EcgSignals")


class _StubEcgSignals:
    def __init__(self, signals=None, fields=None):
        self.signals = signals
        self.fields = fields or {}
        self.leads = list((self.fields or {}).get("sig_name", []))
        self.fs = int((self.fields or {}).get("fs", 500) or 500)


_ECG_SIGNALS_MODULE.EcgSignals = _StubEcgSignals
sys.modules.setdefault("data.EcgSignals", _ECG_SIGNALS_MODULE)

_AGENT_RUNNER_MODULE = ModuleType("agent_runner")


def _stub_process_sample(task):
    return _AGENT_TASK_RESULT_CLASS(
        iteration=task.iteration,
        result_entry={
            "sample_id": task.sample.get("sample_id", "upload-test"),
            "question": task.sample["question"],
            "question_type": task.sample.get("question_type", "user-upload"),
            "template_answer": task.sample.get("template_answer", []),
            "answer": [],
            "final_res": "stubbed result",
            "ecg_tool_routing_decision": {"reason": "Upload pipeline reached real process_sample entry."},
        },
        agent_memories={
            "data_analysis_agent": [
                {
                    "observations": "Measured QRS duration: 96 ms. Rhythm appears regular.",
                    "action_output": "Final answer: stubbed result supported by measured QRS duration and regular rhythm.",
                }
            ]
        },
    )


_AGENT_RUNNER_MODULE.process_sample = _stub_process_sample
sys.modules.setdefault("agent_runner", _AGENT_RUNNER_MODULE)

from thread_executor import AgentTaskResult as _AGENT_TASK_RESULT_CLASS
from webapp import ECGWebService, SampleRequest, create_app
from webapp_support import (
    DEFAULT_DATASET_DIR,
    SnapshotPublisher,
    SnapshotRuntimeStore,
    _prepare_memory_summary_inputs,
    build_evidence_payload,
    load_uploaded_ecg_csv,
)

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - depends on runtime packages
    TestClient = None


def _fake_ecg(n_samples=100, n_leads=12):
    signals = np.zeros((n_samples, n_leads), dtype=float)
    fields = {"sig_name": [f"L{i+1}" for i in range(n_leads)], "fs": 500}
    return _StubEcgSignals(signals=signals, fields=fields)


class SnapshotWebTests(unittest.TestCase):
    def _write_dataset(self, root: Path):
        rows = [
            {
                "sample_id": 1001,
                "question_id": 11,
                "template_id": 1,
                "question_type": "single-verify",
                "question": "Is this ECG normal?",
                "template_answer": ["yes", "no"],
                "answer": ["yes"],
                "ecg_path": ["fake/path/1"],
            },
            {
                "sample_id": 2002,
                "question_id": 22,
                "template_id": 43,
                "question_type": "comparison_consecutive-verify",
                "question": "Has the rhythm changed?",
                "template_answer": ["no", "yes"],
                "answer": ["no"],
                "ecg_path": ["fake/path/2a", "fake/path/2b"],
            },
        ]
        (root / "00000.json").write_text(json.dumps(rows), encoding="utf-8")

    def _write_results(self, path: Path):
        path.write_text(
            json.dumps(
                [
                    {
                        "sample_id": 1001,
                        "question_id": 11,
                        "template_id": 1,
                        "question_type": "single-verify",
                        "question": "Is this ECG normal?",
                        "template_answer": ["yes", "no"],
                        "answer": ["yes"],
                        "final_res": "yes",
                        "is_correct": True,
                        "ecg_tool_routing_decision": {"reason": "Question favors morphology evidence."},
                    },
                    {
                        "sample_id": 2002,
                        "question_id": 22,
                        "template_id": 43,
                        "question_type": "comparison_consecutive-verify",
                        "question": "Has the rhythm changed?",
                        "template_answer": ["no", "yes"],
                        "answer": ["no"],
                        "final_res": "no",
                        "is_correct": True,
                        "ecg_tool_routing_decision": {"reason": "Question asks for structured evidence."},
                    },
                ]
            ),
            encoding="utf-8",
        )

    def test_build_evidence_degrades_without_memory(self):
        result_entry = {
            "final_res": "yes",
            "ecg_tool_routing_decision": {"reason": "Question favors morphology evidence."},
        }
        evidence, degradation = build_evidence_payload(result_entry, agent_memories=None)

        self.assertEqual(evidence["summary"], "未生成可信依据摘要。")
        self.assertEqual(evidence["bullets"], [])
        self.assertEqual(degradation, "memory_missing")

    def test_prepare_memory_summary_inputs_filters_prompt_noise(self):
        snippets = _prepare_memory_summary_inputs(
            {
                "data_analysis_agent": [
                    {"task": "This is the question you need to answer: long prompt that should be filtered."},
                    {
                        "observations": "Measured PR interval: 180 ms. No dropped beats observed.",
                        "action_output": "Final answer: first degree av block supported by measured PR interval prolongation.",
                    },
                ]
            }
        )
        self.assertEqual(len(snippets), 2)
        self.assertEqual(snippets[0]["source"], "action_output")
        self.assertEqual(snippets[1]["source"], "observations")
        self.assertTrue(all("question you need to answer" not in item["text"] for item in snippets))

    def test_build_evidence_uses_ai_summary_from_agent_memory(self):
        result_entry = {
            "final_res": "yes",
            "question": "Is there first degree AV block?",
            "ecg_tool_routing_decision": {"reason": "Question favors morphology evidence."},
        }
        with patch(
            "webapp_support._call_evidence_summary_model",
            return_value={
                "summary": "PR interval prolongation supports first degree AV block.",
                "bullets": [
                    "Measured PR interval: 180 ms.",
                    "No dropped beats observed.",
                ],
                "confidence": "high",
            },
        ):
            evidence, degradation = build_evidence_payload(
                result_entry,
                agent_memories={
                    "data_analysis_agent": [
                        {"task": "This is the question you need to answer: long prompt that should be filtered."},
                        {
                            "observations": "Measured PR interval: 180 ms. No dropped beats observed.",
                            "action_output": "Final answer: first degree av block supported by measured PR interval prolongation.",
                        },
                    ]
                },
            )

        self.assertEqual(degradation, None)
        self.assertEqual(evidence["summary"], "PR interval prolongation supports first degree AV block.")
        self.assertNotIn("Question favors morphology evidence.", evidence["summary"])
        self.assertTrue(any("Measured PR interval" in bullet for bullet in evidence["bullets"]))

    def test_build_evidence_reports_missing_extractable_memory(self):
        evidence, degradation = build_evidence_payload(
            {"final_res": "yes"},
            agent_memories={
                "data_analysis_agent": [
                    {"task": "This is the question you need to answer: prompt only."},
                    {"plan": "Facts survey and plan of action."},
                ]
            },
        )

        self.assertEqual(evidence["summary"], "未生成可信依据摘要。")
        self.assertEqual(evidence["bullets"], [])
        self.assertEqual(degradation, "memory_no_extractable_evidence")

    def test_build_evidence_reports_ai_summary_failure(self):
        with patch("webapp_support._call_evidence_summary_model", side_effect=RuntimeError("boom")):
            evidence, degradation = build_evidence_payload(
                {"question": "Is rhythm regular?", "final_res": "yes"},
                agent_memories={
                    "data_analysis_agent": [
                        {
                            "observations": "Measured RR interval: 820 ms.",
                            "action_output": "Final answer: rhythm is regular.",
                        }
                    ]
                },
            )

        self.assertEqual(evidence["summary"], "未生成可信依据摘要。")
        self.assertEqual(evidence["bullets"], [])
        self.assertEqual(degradation, "summary_generation_failed")

    def test_build_evidence_reports_ai_summary_invalid_payload(self):
        with patch(
            "webapp_support._call_evidence_summary_model",
            return_value={"summary": "facts survey plan", "bullets": ["task"], "confidence": "medium"},
        ):
            evidence, degradation = build_evidence_payload(
                {"question": "Is rhythm regular?", "final_res": "yes"},
                agent_memories={
                    "data_analysis_agent": [
                        {
                            "observations": "Measured RR interval: 820 ms.",
                            "action_output": "Final answer: rhythm is regular.",
                        }
                    ]
                },
            )

        self.assertEqual(evidence["summary"], "未生成可信依据摘要。")
        self.assertEqual(evidence["bullets"], [])
        self.assertEqual(degradation, "summary_generation_invalid")

    def test_snapshot_publish_builds_complete_payload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            self._write_dataset(dataset_dir)
            result_file = root / "results.json"
            self._write_results(result_file)
            mem_file = root / "memories.json"
            mem_file.write_text(
                json.dumps(
                    {
                        "1001": {"data_analysis_agent": [{"content": "Final answer: yes."}]},
                        "2002": {"data_analysis_agent": [{"content": "Final answer: no based on measured intervals."}]},
                    }
                ),
                encoding="utf-8",
            )
            snapshot_root = root / "web_snapshots"
            publisher = SnapshotPublisher(dataset_dir=str(dataset_dir), snapshot_root=str(snapshot_root))
            publisher.sample_store._load_ecg_data = lambda paths: [_fake_ecg() for _ in paths]

            with patch(
                "webapp_support._call_evidence_summary_model",
                side_effect=[
                    {
                        "summary": "ECG evidence supports yes.",
                        "bullets": ["Final answer: yes."],
                        "confidence": "high",
                    },
                    {
                        "summary": "Measured intervals support no rhythm change.",
                        "bullets": ["Final answer: no based on measured intervals."],
                        "confidence": "high",
                    },
                ],
            ):
                publish_result = publisher.publish(
                    result_file=str(result_file),
                    agent_mem_file=str(mem_file),
                    version="20260506T153000Z",
                )

            self.assertEqual(publish_result["sample_count"], 2)
            manifest = json.loads((snapshot_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["current_version"], "20260506T153000Z")
            snapshot = json.loads((snapshot_root / "snapshots" / "20260506T153000Z" / "snapshot.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["sample_count"], 2)
            self.assertEqual([item["name"] for item in snapshot["question_types"]], ["comparison_consecutive-verify", "single-verify"])
            sample_map = {item["sample_id"]: item for item in snapshot["samples"]}
            self.assertEqual(len(sample_map["1001"]["ecg_refs"]), 1)
            self.assertEqual(len(sample_map["2002"]["ecg_refs"]), 2)
            self.assertEqual(sample_map["1001"]["evidence"]["summary"], "ECG evidence supports yes.")

    def test_manifest_is_not_switched_when_publish_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            self._write_dataset(dataset_dir)
            result_file = root / "results.json"
            self._write_results(result_file)
            snapshot_root = root / "web_snapshots"
            publisher = SnapshotPublisher(dataset_dir=str(dataset_dir), snapshot_root=str(snapshot_root))
            publisher.sample_store._load_ecg_data = lambda paths: [_fake_ecg() for _ in paths]
            publisher.publish(result_file=str(result_file), version="v1")

            broken_publisher = SnapshotPublisher(dataset_dir=str(dataset_dir), snapshot_root=str(snapshot_root))
            def _raise_on_load(paths):
                del paths
                raise RuntimeError("boom")
            broken_publisher.sample_store._load_ecg_data = _raise_on_load

            with self.assertRaises(RuntimeError):
                broken_publisher.publish(result_file=str(result_file), version="v2")

            manifest = json.loads((snapshot_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["current_version"], "v1")

    def test_runtime_store_reads_snapshot_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            snapshot_root = root / "web_snapshots"
            version_dir = snapshot_root / "snapshots" / "v1"
            images_dir = version_dir / "images"
            images_dir.mkdir(parents=True)
            (images_dir / "1001_0.png").write_bytes(b"png")
            (images_dir / "2002_0.png").write_bytes(b"png")
            (images_dir / "2002_1.png").write_bytes(b"png")

            snapshot_payload = {
                "version": "v1",
                "created_at": "2026-05-06T15:30:00Z",
                "dataset_dir": DEFAULT_DATASET_DIR,
                "question_types": [
                    {"name": "single-verify", "count": 1},
                    {"name": "comparison_consecutive-verify", "count": 1},
                ],
                "sample_count": 2,
                "samples": [
                    {
                        "sample_id": "1001",
                        "question_id": "11",
                        "template_id": "1",
                        "question_type": "single-verify",
                        "question": "Is this ECG normal?",
                        "choices": ["yes", "no"],
                        "predicted_answer": "yes",
                        "gold_answers": ["yes"],
                        "is_correct": True,
                        "evidence": {"summary": "ok", "bullets": ["a"]},
                        "ecg_refs": [{"title": "ECG 1", "path": "images/1001_0.png"}],
                    },
                    {
                        "sample_id": "2002",
                        "question_id": "22",
                        "template_id": "43",
                        "question_type": "comparison_consecutive-verify",
                        "question": "Has the rhythm changed?",
                        "choices": ["no", "yes"],
                        "predicted_answer": "no",
                        "gold_answers": ["no"],
                        "is_correct": True,
                        "evidence": {"summary": "ok", "bullets": ["b"]},
                        "ecg_refs": [
                            {"title": "ECG 1", "path": "images/2002_0.png"},
                            {"title": "ECG 2", "path": "images/2002_1.png"},
                        ],
                    },
                ],
            }
            (version_dir / "snapshot.json").write_text(json.dumps(snapshot_payload), encoding="utf-8")
            (snapshot_root / "manifest.json").write_text(json.dumps({"current_version": "v1"}), encoding="utf-8")

            store = SnapshotRuntimeStore(snapshot_root=str(snapshot_root))
            options = store.get_options()
            self.assertEqual(options["version"], "v1")
            row = store.get_random_sample("comparison_consecutive-verify")
            payload = store.build_frontend_payload(row)
            self.assertEqual(payload["snapshot_version"], "v1")
            self.assertEqual(len(payload["ecg_images"]), 2)
            token = payload["ecg_images"][0]["image_url"].split("/api/ecg/image/")[1]
            self.assertTrue(store.resolve_image(token).exists())

    def test_web_service_uses_snapshot_runtime(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            snapshot_root = root / "web_snapshots"
            version_dir = snapshot_root / "snapshots" / "v2"
            images_dir = version_dir / "images"
            images_dir.mkdir(parents=True)
            (images_dir / "1001_0.png").write_bytes(b"png")
            (version_dir / "snapshot.json").write_text(
                json.dumps(
                    {
                        "version": "v2",
                        "created_at": "2026-05-06T16:00:00Z",
                        "dataset_dir": DEFAULT_DATASET_DIR,
                        "question_types": [{"name": "single-verify", "count": 1}],
                        "sample_count": 1,
                        "samples": [
                            {
                                "sample_id": "1001",
                                "question_id": "11",
                                "template_id": "1",
                                "question_type": "single-verify",
                                "question": "Is this ECG normal?",
                                "choices": ["yes", "no"],
                                "predicted_answer": "yes",
                                "gold_answers": ["yes"],
                                "is_correct": True,
                                "evidence": {"summary": "ok", "bullets": ["a"]},
                                "ecg_refs": [{"title": "ECG 1", "path": "images/1001_0.png"}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (snapshot_root / "manifest.json").write_text(json.dumps({"current_version": "v2"}), encoding="utf-8")

            service = ECGWebService(snapshot_root=str(snapshot_root))
            options = service.get_options()
            self.assertEqual(options["sample_count"], 1)
            payload = service.next_sample(SampleRequest(question_type="single-verify"))
            self.assertEqual(payload["predicted_answer"], "yes")
            self.assertEqual(payload["snapshot_version"], "v2")
            token = payload["ecg_images"][0]["image_url"].split("/api/ecg/image/")[1]
            self.assertTrue(service.resolve_image(token).exists())

    def test_runtime_store_raises_when_manifest_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                SnapshotRuntimeStore(snapshot_root=str(Path(tmp_dir) / "web_snapshots"))

    def test_load_uploaded_ecg_csv_supports_header_row(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "uploaded.csv"
            csv_path.write_text("I,II,III\n0.1,0.2,0.3\n0.4,0.5,0.6\n", encoding="utf-8")

            ecg = load_uploaded_ecg_csv(csv_path, fs=250)

            self.assertEqual(ecg.fs, 250)
            self.assertEqual(ecg.leads, ["I", "II", "III"])
            self.assertEqual(ecg.signals.shape, (2, 3))

    @unittest.skipIf(TestClient is None, "fastapi test client unavailable")
    def test_upload_analyze_returns_renderable_payload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            snapshot_root = root / "web_snapshots"
            version_dir = snapshot_root / "snapshots" / "v1"
            images_dir = version_dir / "images"
            images_dir.mkdir(parents=True)
            (images_dir / "1001_0.png").write_bytes(b"png")
            (version_dir / "snapshot.json").write_text(
                json.dumps(
                    {
                        "version": "v1",
                        "created_at": "2026-05-06T16:00:00Z",
                        "dataset_dir": DEFAULT_DATASET_DIR,
                        "question_types": [{"name": "single-verify", "count": 1}],
                        "sample_count": 1,
                        "samples": [
                            {
                                "sample_id": "1001",
                                "question_id": "11",
                                "template_id": "1",
                                "question_type": "single-verify",
                                "question": "Is this ECG normal?",
                                "choices": ["yes", "no"],
                                "predicted_answer": "yes",
                                "gold_answers": ["yes"],
                                "is_correct": True,
                                "evidence": {"summary": "ok", "bullets": ["a"]},
                                "ecg_refs": [{"title": "ECG 1", "path": "images/1001_0.png"}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (snapshot_root / "manifest.json").write_text(json.dumps({"current_version": "v1"}), encoding="utf-8")

            app = create_app(snapshot_root=str(snapshot_root), upload_root=str(root / "web_uploads"))
            client = TestClient(app)
            with patch(
                "webapp_support._call_evidence_summary_model",
                return_value={
                    "summary": "QRS duration and rhythm regularity support the stubbed result.",
                    "bullets": ["Measured QRS duration: 96 ms.", "Rhythm appears regular."],
                    "confidence": "high",
                },
            ):
                response = client.post(
                    "/api/upload/analyze",
                    data={"question": "What is the main abnormality?", "fs": "500"},
                    files={"ecg_file": ("uploaded.csv", b"I,II\n0.1,0.2\n0.3,0.4\n", "text/csv")},
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["mode"], "upload")
            self.assertEqual(payload["predicted_answer"], "stubbed result")
            self.assertEqual(payload["upload"]["filename"], "uploaded.csv")
            self.assertEqual(len(payload["ecg_images"]), 1)
            self.assertEqual(payload["evidence"]["summary"], "QRS duration and rhythm regularity support the stubbed result.")
            self.assertTrue(any("Measured QRS duration" in bullet for bullet in payload["evidence"]["bullets"]))

    def test_frontend_template_hides_ids_and_choices(self):
        html = Path("web/index.html").read_text(encoding="utf-8")
        self.assertNotIn("sample_id:", html)
        self.assertNotIn("question_id:", html)
        self.assertNotIn("template_id:", html)
        self.assertNotIn("payload.choices", html)
        self.assertIn("自主上传", html)
        self.assertIn("/api/upload/analyze", html)


if __name__ == "__main__":
    unittest.main()
