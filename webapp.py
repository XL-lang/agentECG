import argparse
import importlib.util
import os
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any

from webapp_support import (
    DEFAULT_DATASET_DIR,
    DEFAULT_SNAPSHOT_ROOT,
    DEFAULT_UPLOAD_ROOT,
    SnapshotPublisher,
    SnapshotRuntimeStore,
    build_evidence_payload,
    load_text_file,
    load_uploaded_ecg_csv,
    render_ecg_image,
)

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - exercised only in missing-runtime envs
    FastAPI = None
    HTTPException = RuntimeError
    Request = None
    FileResponse = None
    HTMLResponse = None

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    _FASTAPI_IMPORT_ERROR = exc
else:
    _FASTAPI_IMPORT_ERROR = None


_MULTIPART_AVAILABLE = importlib.util.find_spec("multipart") is not None


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_INDEX_PATH = PROJECT_ROOT / "web" / "index.html"


class SampleRequest(BaseModel):
    question_type: str | None = None


class ECGWebService:
    def __init__(
        self,
        snapshot_root: str = DEFAULT_SNAPSHOT_ROOT,
        upload_root: str = DEFAULT_UPLOAD_ROOT,
    ):
        self.snapshot_root = snapshot_root
        self.snapshot_store = SnapshotRuntimeStore(snapshot_root=snapshot_root)
        self.upload_root = Path(upload_root)
        self.upload_root.mkdir(parents=True, exist_ok=True)

    def get_options(self) -> dict[str, Any]:
        self.snapshot_store.reload()
        return self.snapshot_store.get_options()

    def next_sample(self, request: SampleRequest) -> dict[str, Any]:
        self.snapshot_store.reload()
        sample_row = self.snapshot_store.get_random_sample(question_type=request.question_type)
        return self.snapshot_store.build_frontend_payload(sample_row)

    def resolve_image(self, token: str) -> Path:
        return self.snapshot_store.resolve_image(token)

    def resolve_upload_image(self, token: str) -> Path:
        path = self.upload_root / token
        if not path.exists():
            raise KeyError(token)
        return path

    def create_upload_payload(
        self,
        *,
        question: str,
        ecg_file_path: Path,
        original_filename: str,
        fs: int,
    ) -> dict[str, Any]:
        uploaded_ecg = load_uploaded_ecg_csv(ecg_file_path, fs=fs)
        sample_id = f"upload-{uuid.uuid4().hex[:12]}"
        sample = {
            "sample_id": sample_id,
            "question_id": "",
            "template_id": "",
            "question_type": "user-upload",
            "question": question,
            "template_answer": [],
            "answer": [],
            "ecg_path": [str(ecg_file_path)],
            "ecg_datas": [uploaded_ecg],
        }
        preview_name = f"{sample_id}.png"
        preview_path = self.upload_root / preview_name
        render_ecg_image(uploaded_ecg, preview_path, title="Uploaded ECG")

        from agent_runner import process_sample
        from thread_executor import AgentExecutionOptions, AgentTask

        task = AgentTask(
            iteration=1,
            sample=sample,
            options=AgentExecutionOptions(
                save_agent_mem=False,
                agent_mem_file=str(self.upload_root / f"{sample_id}_memories.json"),
                run_mode="eval",
            ),
        )
        task_result = process_sample(task)
        if task_result.error:
            raise RuntimeError(task_result.error)
        if task_result.result_entry is None:
            raise RuntimeError("Upload inference returned no result entry.")

        result_entry = task_result.result_entry
        evidence, _degradation_reason = build_evidence_payload(
            result_entry,
            agent_memories=task_result.agent_memories,
        )
        predicted_answer = result_entry.get("final_res")

        return {
            "mode": "upload",
            "question_type": "user-upload",
            "question": question,
            "predicted_answer": predicted_answer,
            "gold_answers": [],
            "is_correct": False,
            "evidence": evidence,
            "ecg_images": [
                {
                    "index": 0,
                    "title": original_filename or "Uploaded ECG",
                    "image_url": f"/api/upload/image/{preview_name}",
                }
            ],
            "upload": {
                "filename": original_filename,
                "fs": int(fs),
                "n_leads": int(getattr(uploaded_ecg.signals, "shape", [0, 0])[1]),
                "n_samples": int(getattr(uploaded_ecg.signals, "shape", [0, 0])[0]),
            },
            "raw_result": {
                "final_res": predicted_answer,
                "routing_reason": str(result_entry.get("ecg_tool_routing_decision", {}).get("reason", "")).strip(),
            },
        }


def create_app(
    snapshot_root: str = DEFAULT_SNAPSHOT_ROOT,
    upload_root: str = DEFAULT_UPLOAD_ROOT,
) -> "FastAPI":
    if FastAPI is None:  # pragma: no cover - depends on runtime packages
        raise RuntimeError(
            "FastAPI is not installed. Install `fastapi` and `uvicorn` to run the ECG web app."
        ) from _FASTAPI_IMPORT_ERROR

    service = ECGWebService(snapshot_root=snapshot_root, upload_root=upload_root)
    app = FastAPI(title="agentECG Web UI")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(load_text_file(WEB_INDEX_PATH))

    @app.get("/api/options")
    def get_options() -> dict[str, Any]:
        try:
            return service.get_options()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (LookupError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/sample/next")
    def next_sample(request: SampleRequest) -> dict[str, Any]:
        try:
            return service.next_sample(request)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/ecg/image/{token:path}")
    def ecg_image(token: str) -> FileResponse:
        try:
            image_path = service.resolve_image(token)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"ECG image not found: {token}") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return FileResponse(image_path, media_type="image/png")

    @app.get("/api/upload/image/{token:path}")
    def upload_image(token: str) -> FileResponse:
        try:
            image_path = service.resolve_upload_image(token)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Upload ECG image not found: {token}") from exc
        return FileResponse(image_path, media_type="image/png")

    @app.post("/api/upload/analyze")
    async def upload_analyze(request: Request) -> dict[str, Any]:
        if not _MULTIPART_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail='Upload support requires "python-multipart" to be installed.',
            )
        form = await request.form()
        question = str(form.get("question", "")).strip()
        fs = int(form.get("fs", 500) or 500)
        ecg_file = form.get("ecg_file")
        if not question:
            raise HTTPException(status_code=400, detail="Question must not be empty.")
        if ecg_file is None or getattr(ecg_file, "filename", None) is None:
            raise HTTPException(status_code=400, detail="ECG upload file is required.")
        if not ecg_file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV ECG uploads are supported right now.")
        if fs <= 0:
            raise HTTPException(status_code=400, detail="Sampling rate must be positive.")

        upload_dir = Path(tempfile.mkdtemp(prefix="upload_", dir=str(service.upload_root)))
        target_path = upload_dir / ecg_file.filename
        payload = await ecg_file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded ECG file is empty.")
        target_path.write_bytes(payload)

        try:
            return service.create_upload_payload(
                question=question,
                ecg_file_path=target_path,
                original_filename=ecg_file.filename,
                fs=fs,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            detail = str(exc).strip() or traceback.format_exc()
            raise HTTPException(status_code=500, detail=detail) from exc

    return app


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description="agentECG web snapshot tooling")
    subparsers = parser.add_subparsers(dest="command")

    publish_parser = subparsers.add_parser("publish", help="Build and publish a new offline web snapshot")
    publish_parser.add_argument("--result-file", required=True)
    publish_parser.add_argument("--agent-mem-file")
    publish_parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    publish_parser.add_argument("--snapshot-root", default=DEFAULT_SNAPSHOT_ROOT)
    publish_parser.add_argument("--version")

    args = parser.parse_args()
    if args.command == "publish":
        publisher = SnapshotPublisher(dataset_dir=args.dataset_dir, snapshot_root=args.snapshot_root)
        result = publisher.publish(
            result_file=args.result_file,
            agent_mem_file=args.agent_mem_file,
            version=args.version,
        )
        print(result)
        return

    if _FASTAPI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "FastAPI is not installed. Install `fastapi` and `uvicorn`, then run `uvicorn webapp:app --reload`."
        ) from _FASTAPI_IMPORT_ERROR

    import uvicorn

    uvicorn.run(
        "webapp:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if FastAPI is not None:  # pragma: no branch
    app = create_app(
        snapshot_root=os.environ.get("AGENTECG_WEB_SNAPSHOT_ROOT", DEFAULT_SNAPSHOT_ROOT),
        upload_root=os.environ.get("AGENTECG_WEB_UPLOAD_ROOT", DEFAULT_UPLOAD_ROOT),
    )


if __name__ == "__main__":
    main()
