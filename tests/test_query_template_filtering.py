import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
import sys

import numpy as np

_TORCH_MODULE = ModuleType("torch")
_TORCH_UTILS_MODULE = ModuleType("torch.utils")
_TORCH_UTILS_DATA_MODULE = ModuleType("torch.utils.data")
_WFDB_MODULE = ModuleType("wfdb")
_SKLEARN_MODULE = ModuleType("sklearn")
_SKLEARN_PREPROCESSING_MODULE = ModuleType("sklearn.preprocessing")
_INCEPTION_TIME_MODULE = ModuleType("models.pytorch_inception.inception_time")
_ECG_SIGNALS_MODULE = ModuleType("data.EcgSignals")


class _IterableDataset:
    pass


class _StandardScaler:
    pass


class _Cuda:
    @staticmethod
    def is_available():
        return False


class _StubEcgSignals:
    def __init__(self, signals=None, fields=None):
        self.signals = signals
        self.fields = fields


_TORCH_UTILS_DATA_MODULE.IterableDataset = _IterableDataset
_TORCH_UTILS_MODULE.data = _TORCH_UTILS_DATA_MODULE
_TORCH_MODULE.utils = _TORCH_UTILS_MODULE
_TORCH_MODULE.cuda = _Cuda()
_TORCH_MODULE.device = lambda value: value
_SKLEARN_PREPROCESSING_MODULE.StandardScaler = _StandardScaler
_INCEPTION_TIME_MODULE.create_inception_time_model = lambda *args, **kwargs: None
_ECG_SIGNALS_MODULE.EcgSignals = _StubEcgSignals

sys.modules.setdefault("torch", _TORCH_MODULE)
sys.modules.setdefault("torch.utils", _TORCH_UTILS_MODULE)
sys.modules.setdefault("torch.utils.data", _TORCH_UTILS_DATA_MODULE)
sys.modules.setdefault("wfdb", _WFDB_MODULE)
sys.modules.setdefault("sklearn", _SKLEARN_MODULE)
sys.modules.setdefault("sklearn.preprocessing", _SKLEARN_PREPROCESSING_MODULE)
sys.modules.setdefault("models.pytorch_inception.inception_time", _INCEPTION_TIME_MODULE)
sys.modules.setdefault("data.EcgSignals", _ECG_SIGNALS_MODULE)

from data.EcgQaData import EcgQaDataset
from dataset.config import ALLOWED_QUERY_TEMPLATE_IDS
from models.pytorch_inception.ecgqa_classifier import ECGQAClassifierManager


class _DummyECG:
    def __init__(self, length=1000, leads=12):
        self.signals = np.zeros((length, leads), dtype=float)


class QueryTemplateFilteringTests(unittest.TestCase):
    SINGLE_ECG_ALLOWED = (8, 9, 18, 19, 22, 25, 38)
    TWO_ECG_ALLOWED = (47, 48, 54, 61, 68, 70)

    def _write_dataset(self, root: Path):
        samples = [
            {
                "sample_id": 1,
                "template_id": 8,
                "question_type": "single-query",
                "ecg_path": [],
            },
            {
                "sample_id": 2,
                "template_id": 10,
                "question_type": "single-query",
                "ecg_path": [],
            },
            {
                "sample_id": 3,
                "template_id": 61,
                "question_type": "comparison_irrelevant-query",
                "ecg_path": [],
            },
            {
                "sample_id": 4,
                "template_id": 12,
                "question_type": "single-verify",
                "ecg_path": [],
            },
        ]
        (root / "00000.json").write_text(json.dumps(samples), encoding="utf-8")

    def test_dataset_filters_query_templates_but_keeps_non_query(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_dataset(root)
            dataset = EcgQaDataset(str(root), lazy=True)
            rows = list(dataset)

        self.assertEqual([row["sample_id"] for row in rows], [1, 3, 4])
        self.assertEqual(
            [row["template_id"] for row in rows if "query" in row["question_type"]],
            [8, 61],
        )

    def test_dataset_with_query_question_type_only_uses_whitelist(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_dataset(root)
            dataset = EcgQaDataset(
                str(root),
                question_types=["single-query", "comparison_irrelevant-query"],
                lazy=True,
            )
            rows = list(dataset)

        self.assertEqual([row["template_id"] for row in rows], [8, 61])

    def test_classifier_manager_exposes_only_allowed_single_ecg_query_templates(self):
        manager = ECGQAClassifierManager([_DummyECG()])

        self.assertEqual(tuple(manager.query_ids), self.SINGLE_ECG_ALLOWED)
        self.assertEqual(set(manager.question.keys()), set(self.SINGLE_ECG_ALLOWED))

    def test_classifier_manager_exposes_only_allowed_two_ecg_query_templates(self):
        manager = ECGQAClassifierManager([_DummyECG(), _DummyECG()])

        self.assertEqual(tuple(manager.query_ids), self.TWO_ECG_ALLOWED)
        self.assertEqual(set(manager.question.keys()), set(self.TWO_ECG_ALLOWED))
        self.assertEqual(
            set(manager.query_ids),
            set(ALLOWED_QUERY_TEMPLATE_IDS).intersection(manager.question.keys()),
        )

    def test_query_whitelist_excludes_noise_templates(self):
        self.assertNotIn(33, ALLOWED_QUERY_TEMPLATE_IDS)
        self.assertNotIn(34, ALLOWED_QUERY_TEMPLATE_IDS)
        self.assertEqual(len(ALLOWED_QUERY_TEMPLATE_IDS), 13)


if __name__ == "__main__":
    unittest.main()
