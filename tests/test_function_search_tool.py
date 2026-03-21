import os
import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


class _Tool:
    pass


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("extra_body", {}).get("enable_search") is True:
            return _FakeCompletion("web answer")
        return _FakeCompletion("kb answer")


class _FakeReasonClient:
    def __init__(self):
        self.completions = _FakeCompletions()
        self.chat = type("Chat", (), {"completions": self.completions})()


def _load_search_tool_module():
    utils_module = ModuleType("utils")
    utils_module.reason_client = _FakeReasonClient()

    smolagents_module = ModuleType("smolagents")
    smolagents_module.Tool = _Tool

    sys.modules["utils"] = utils_module
    sys.modules["smolagents"] = smolagents_module

    module_path = Path(__file__).resolve().parents[1] / "mAgents" / "function_search_tool.py"
    spec = spec_from_file_location("test_function_search_tool_module", module_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakePoint:
    def __init__(self, payload):
        self.payload = payload


class SearchToolTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_search_tool_module()
        self.tool = self.module.SearchTool()
        self.reason_client = self.module.reason_client

    def test_missing_config_falls_back_to_web_search(self):
        with patch.dict(os.environ, {}, clear=True):
            result = self.tool.forward("What is LVH?")

        self.assertTrue(result.startswith("[source=web_search]\n"))
        self.assertEqual(self.reason_client.completions.calls[0]["extra_body"], {"enable_search": True})
        self.assertIsInstance(result, str)

    def test_probe_failure_falls_back_to_web_search(self):
        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "http://localhost:6333",
                "QDRANT_COLLECTION": "ecg",
            },
            clear=True,
        ), patch.object(self.module, "_probe_qdrant", return_value=False):
            result = self.tool.forward("What is LVH?")

        self.assertTrue(result.startswith("[source=web_search]\n"))
        self.assertEqual(self.reason_client.completions.calls[0]["extra_body"], {"enable_search": True})

    def test_missing_qdrant_client_falls_back_to_web_search(self):
        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "http://localhost:6333",
                "QDRANT_COLLECTION": "ecg",
            },
            clear=True,
        ), patch.object(self.module, "_probe_qdrant", return_value=True), patch.object(
            self.module,
            "_get_qdrant_client_components",
            side_effect=ImportError("missing qdrant client"),
        ):
            result = self.tool.forward("What is LVH?")

        self.assertTrue(result.startswith("[source=web_search]\n"))
        self.assertEqual(self.reason_client.completions.calls[0]["extra_body"], {"enable_search": True})

    def test_qdrant_query_exception_falls_back_to_web_search(self):
        class _BrokenQdrantClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def query_points(self, **kwargs):
                raise RuntimeError("boom")

        fake_models = type(
            "Models",
            (),
            {"Document": lambda **kwargs: kwargs},
        )

        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "http://localhost:6333",
                "QDRANT_COLLECTION": "ecg",
            },
            clear=True,
        ), patch.object(self.module, "_probe_qdrant", return_value=True), patch.object(
            self.module,
            "_get_qdrant_client_components",
            return_value=(_BrokenQdrantClient, fake_models),
        ):
            result = self.tool.forward("What is LVH?")

        self.assertTrue(result.startswith("[source=web_search]\n"))
        self.assertEqual(self.reason_client.completions.calls[0]["extra_body"], {"enable_search": True})

    def test_successful_qdrant_query_returns_qdrant_source(self):
        class _WorkingQdrantClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def query_points(self, **kwargs):
                return type(
                    "Response",
                    (),
                    {"points": [_FakePoint({"document": "Left ventricular hypertrophy criteria."})]},
                )()

        fake_models = type(
            "Models",
            (),
            {"Document": lambda **kwargs: kwargs},
        )

        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "http://localhost:6333",
                "QDRANT_COLLECTION": "ecg",
                "QDRANT_QUERY_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
            },
            clear=True,
        ), patch.object(self.module, "_probe_qdrant", return_value=True), patch.object(
            self.module,
            "_get_qdrant_client_components",
            return_value=(_WorkingQdrantClient, fake_models),
        ):
            result = self.tool.forward("What is LVH?")

        self.assertTrue(result.startswith("[source=qdrant]\n"))
        self.assertEqual(len(self.reason_client.completions.calls), 1)
        self.assertNotIn("extra_body", self.reason_client.completions.calls[0])

    def test_probe_qdrant_uses_readyz_then_healthz(self):
        config = self.module._QdrantConfig(
            backend="qdrant",
            url="http://localhost:6333",
            api_key="secret",
            collection="ecg",
            top_k=5,
            query_model="sentence-transformers/all-MiniLM-L6-v2",
            timeout_seconds=2.0,
        )

        class _Response:
            def __init__(self, status):
                self.status = status

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        captured_urls = []

        def _fake_urlopen(req, timeout):
            captured_urls.append((req.full_url, timeout, req.headers.get("Api-key")))
            if req.full_url.endswith("/readyz"):
                raise self.module.error.URLError("not ready")
            return _Response(200)

        with patch.object(self.module.request, "urlopen", side_effect=_fake_urlopen):
            ok = self.module._probe_qdrant(config)

        self.assertTrue(ok)
        self.assertEqual(captured_urls[0][0], "http://localhost:6333/readyz")
        self.assertEqual(captured_urls[1][0], "http://localhost:6333/healthz")
        self.assertEqual(captured_urls[1][2], "secret")


if __name__ == "__main__":
    unittest.main()
