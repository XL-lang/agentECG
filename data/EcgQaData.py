import os
import json
import glob
from typing import Iterator, Dict, Any, Optional, List
import torch
from torch.utils.data import IterableDataset
import numpy as np
import scipy.io
from pathlib import Path
import wfdb
import random
from .EcgSignals import EcgSignals
from dataset.config import ecg_qa_types
import shutil
import pandas as pd
import ast
from typing import Dict

class EcgQaDataset(IterableDataset):
    def __init__(self, dataset_dir: str, question_types: Optional[List[str]] = None, sample_limit: Optional[int] = None, shuffle: bool = False, seed: Optional[int] = None):
        """
        PyTorch IterableDataset for ECG-QA data.

        Args:
            dataset_dir (str): Path to the ECG-QA dataset directory.
            question_types (Optional[List[str]]): Filter by question types.
            sample_limit (Optional[int]): Max samples per question type.
            shuffle (bool): If True, randomly shuffle file order and sample order before reading ECG signals (avoids extra work in the heavy rdsamp loop).
            seed (Optional[int]): Random seed for reproducible shuffling.
        """
        super(EcgQaDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.data_files = glob.glob(os.path.join(self.dataset_dir, "*.json"))
        self.shuffle = shuffle
        self._rng = random.Random(seed) if shuffle else None
        self.file_index = 0
        self.current_file_data = []
        self.current_data_index = 0
        self.sample_limit = sample_limit
        self.question_types = question_types
        self.question_types_limit = None
        if (self.question_types is not None)and (self.sample_limit is not None):
            self.question_types_limit = {qt: self.sample_limit for qt in self.question_types}   
        elif (self.sample_limit is not None):
            self.question_types_limit = {}
            for qt in ecg_qa_types:
                self.question_types_limit[qt] = self.sample_limit

        # Shuffle file order once up-front (cheap) to avoid touching heavy signal reads
        if self.shuffle and self.data_files:
            self._rng.shuffle(self.data_files)

    
    def __str__(self):
        """Return a user-friendly string representation"""
        return f"EcgQaDataset(dataset_dir='{self.dataset_dir}', files={len(self.data_files)})"
    
    def __repr__(self):
        """Return a developer-friendly string representation"""
        return f"EcgQaDataset(dataset_dir='{self.dataset_dir}', data_files={len(self.data_files)} files, current_file_index={self.file_index})"
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        while self.file_index < len(self.data_files):
            if self.current_data_index >= len(self.current_file_data):
                # Load next file
                with open(self.data_files[self.file_index], 'r') as f:
                    self.current_file_data = json.load(f)
                # Shuffle samples within the file before any expensive ECG loading
                if self.shuffle and self.current_file_data:
                    self._rng.shuffle(self.current_file_data)
                self.current_data_index = 0
                self.file_index += 1

            while self.current_data_index < len(self.current_file_data):
                sample = self.current_file_data[self.current_data_index]
                if self.question_types is not None:
                    qt = sample.get('question_type')
                    if qt not in self.question_types:
                        self.current_data_index += 1
                        continue
                if self.question_types_limit is not None:
                    qt = sample.get('question_type')
                    if self.question_types_limit[qt] <= 0:
                        self.current_data_index += 1
                        continue
                    self.question_types_limit[qt] -= 1

                self.current_data_index += 1
                ecg_paths = sample.get('ecg_path')
                ecg_datas = []
                for ecg_path in ecg_paths:
                    sigals, fields = wfdb.rdsamp(ecg_path)
                    ecg_data = EcgSignals(signals=sigals, fields=fields)
                    ecg_datas.append(ecg_data)
                sample['ecg_datas'] = ecg_datas
                yield sample

class ECGQAClassificationDataset(IterableDataset):
    def __init__(self, dataset_dir: str, template_id:int,shuffle: bool = False, seed: Optional[int] = None):
        super(ECGQAClassificationDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.data_files = glob.glob(os.path.join(self.dataset_dir, "*.json"))
        self.shuffle = shuffle
        self._rng = random.Random(seed) if shuffle else None
        self.file_index = 0
        self.current_file_data = []
        self.current_data_index = 0
        self.template_id = template_id
        self.classification_info = {
            "input_ecg_num" : 0,
            "output_dim" : 0,
            "output_classes": None,
            "question" : None,

        }
        self._init_classification_info()
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        while self.file_index < len(self.data_files):
            if self.current_data_index >= len(self.current_file_data):
                # Load next file
                with open(self.data_files[self.file_index], 'r') as f:
                    self.current_file_data = json.load(f)
                # Shuffle samples within the file before any expensive ECG loading
                if self.shuffle and self.current_file_data:
                    self._rng.shuffle(self.current_file_data)  # pyright: ignore[reportOptionalMemberAccess]
                self.current_data_index = 0
                self.file_index += 1
            while self.current_data_index < len(self.current_file_data):
                sample = self.current_file_data[self.current_data_index]
                if int(sample["template_id"]) != self.template_id:
                    self.current_data_index += 1
                    continue
                self.current_data_index += 1
                ecg_paths = sample.get('ecg_path')
                ecg_datas = []
                for ecg_path in ecg_paths:
                    sigals, fields = wfdb.rdsamp(ecg_path)
                    ecg_data = EcgSignals(signals=sigals, fields=fields)
                    ecg_datas.append(ecg_data)
                sample['ecg_datas'] = ecg_datas
                sample["template_answer"] = sorted(sample["template_answer"])
            
                yield sample
    def _init_classification_info(self):
        while self.file_index < len(self.data_files):
            if self.current_data_index >= len(self.current_file_data):
                # Load next file
                with open(self.data_files[self.file_index], 'r') as f:
                    self.current_file_data = json.load(f)
                self.current_data_index = 0
                self.file_index += 1
            while self.current_data_index < len(self.current_file_data):
                sample = self.current_file_data[self.current_data_index]
                if int(sample["template_id"]) != self.template_id:
                    self.current_data_index += 1
                    continue
                input_ecg_num = len(sample["ecg_path"])
                output_dim = len(sample["template_answer"])
                output_classes = sample["template_answer"]
                question = sample["question"]
                self.classification_info["input_ecg_num"] += input_ecg_num
                self.classification_info["output_dim"] += output_dim
                self.classification_info["output_classes"] = output_classes
                self.classification_info["question"] = question

                self.current_data_index = 0
                self.file_index = 0
                return

class ECGQAClassificationManager:
    def __init__(self, dataset_dir: str, class_id_path: str="test_tools/ecgqa_info/class_id.csv"):
        self.class_id_df = pd.read_csv(class_id_path)
        ids_list_list = self.class_id_df["ids"].tolist()
        ids_list_list = [ast.literal_eval(id_list) for id_list in ids_list_list]
        self.ids = sorted([int(item) for sublist in ids_list_list for item in sublist])
        self.dataset_dict: Dict[int, ECGQAClassificationDataset] = {}
        for template_id in self.ids:
            self.create_classification_dataset(dataset_dir=dataset_dir, template_id=template_id)
    def create_classification_dataset(self, dataset_dir: str, template_id: int):
        self.dataset_dict[template_id] = ECGQAClassificationDataset(dataset_dir=dataset_dir, template_id=template_id)