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
from dataset.config import ecg_qa_types, is_allowed_ecgqa_sample
import shutil
import pandas as pd
import ast
from typing import List, Dict, Union
from enum import Enum


class QueryType(Enum):
    sample_id_query = "sample_id_query"
    template_id_query = "template_id_query"


class Special_query:
    def __init__(self, query_type: QueryType, query_value: str):
        self.query_type = query_type
        self.query_value = query_value


class EcgQaDataset(IterableDataset):
    def __init__(self, dataset_dir: str, question_types: Optional[List[str]] = None, sample_limit: Optional[int] = None, shuffle: bool = False, seed: Optional[int] = None, lazy: bool = False, special_queries: Optional[List[Special_query]] = None):
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
        self.special_queries = special_queries
        self.lazy = lazy
        if (self.question_types is not None) and (self.sample_limit is not None):
            self.question_types_limit = {
                qt: self.sample_limit for qt in self.question_types}
        elif (self.sample_limit is not None):
            self.question_types_limit = {}
            for qt in ecg_qa_types:
                self.question_types_limit[qt] = self.sample_limit

        # Shuffle file order once up-front (cheap) to avoid touching heavy signal reads
        if self.shuffle and self.data_files and self._rng is not None:
            self._rng.shuffle(self.data_files)

    def __str__(self):
        """Return a user-friendly string representation"""
        return f"EcgQaDataset(dataset_dir='{self.dataset_dir}', files={len(self.data_files)})"

    def __repr__(self):
        """Return a developer-friendly string representation"""
        return f"EcgQaDataset(dataset_dir='{self.dataset_dir}', data_files={len(self.data_files)} files, current_file_index={self.file_index})"

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # 如果有sample_limit，使用改进的shuffle策略：先收集所有符合条件的样本元数据，再shuffle
        if self.sample_limit is not None:
            return self._iter_with_sample_limit()
        else:
            return self._iter_without_sample_limit()

    def _is_sample_eligible(self, sample: Dict[str, Any]) -> bool:
        if not is_allowed_ecgqa_sample(sample):
            return False

        if self.special_queries is not None:
            is_special_query = False
            for special_query in self.special_queries:
                if special_query.query_type == QueryType.sample_id_query:
                    if str(sample.get('sample_id')) == str(special_query.query_value):
                        is_special_query = True
                elif special_query.query_type == QueryType.template_id_query:
                    if str(sample.get('template_id')) == str(special_query.query_value):
                        is_special_query = True
            if not is_special_query:
                return False

        if self.question_types is not None:
            qt = sample.get('question_type')
            if qt not in self.question_types:
                return False

        return True
    
    def _iter_with_sample_limit(self) -> Iterator[Dict[str, Any]]:
        """
        当有sample_limit时的迭代方法：
        1. 先用lazy模式收集所有符合条件的样本元数据（不加载ECG数据，不应用limit）
        2. 收集完成后，对这些元数据进行shuffle
        3. 在shuffle后的数据上应用sample_limit
        4. 在yield时再加载ECG数据
        """
        # 第一步：收集所有符合条件的样本元数据（不应用limit）
        collected_samples = []
        
        # 遍历所有文件，收集样本元数据
        for file_path in self.data_files:
            with open(file_path, 'r') as f:
                file_data = json.load(f)
            
            # 如果shuffle，先对文件内数据进行shuffle
            if self.shuffle and file_data and self._rng is not None:
                self._rng.shuffle(file_data)
            
            for sample in file_data:
                if not self._is_sample_eligible(sample):
                    continue
                
                # 收集样本元数据（不加载ECG数据，不应用limit）
                collected_samples.append(sample)
        
        # 第二步：对所有收集到的样本进行shuffle
        if self.shuffle and collected_samples and self._rng is not None:
            self._rng.shuffle(collected_samples)
        
        # 第三步：在shuffle后的数据上应用sample_limit
        temp_question_types_limit = {}
        if self.question_types_limit is not None:
            temp_question_types_limit = {k: v for k, v in self.question_types_limit.items()}
        
        # 按shuffle后的顺序应用limit并yield，在yield时加载ECG数据
        for sample in collected_samples:
            # 应用question_types_limit
            if temp_question_types_limit is not None:
                qt = sample.get('question_type')
                if qt not in temp_question_types_limit or temp_question_types_limit[qt] <= 0:
                    continue
                temp_question_types_limit[qt] -= 1
            
            # 加载ECG数据并yield
            if self.lazy:
                yield sample
            else:   
                ecg_paths = sample.get('ecg_path')
                ecg_datas = []
                for ecg_path in ecg_paths:
                    sigals, fields = wfdb.rdsamp(ecg_path)
                    ecg_data = EcgSignals(signals=sigals, fields=fields)
                    ecg_datas.append(ecg_data)
                sample['ecg_datas'] = ecg_datas
                yield sample
    
    def _iter_without_sample_limit(self) -> Iterator[Dict[str, Any]]:
        """
        当没有sample_limit时的迭代方法：保持原来的方式
        """
        while self.file_index < len(self.data_files):
            if self.current_data_index >= len(self.current_file_data):
                # Load next file
                with open(self.data_files[self.file_index], 'r') as f:
                    self.current_file_data = json.load(f)
                # Shuffle samples within the file before any expensive ECG loading
                if self.shuffle and self.current_file_data and self._rng is not None:
                    self._rng.shuffle(self.current_file_data)
                self.current_data_index = 0
                self.file_index += 1

            while self.current_data_index < len(self.current_file_data):
                sample = self.current_file_data[self.current_data_index]
                if not self._is_sample_eligible(sample):
                    self.current_data_index += 1
                    continue

                self.current_data_index += 1
                if not self.lazy:
                    ecg_paths = sample.get('ecg_path')
                    ecg_datas = []
                    for ecg_path in ecg_paths:
                        sigals, fields = wfdb.rdsamp(ecg_path)
                        ecg_data = EcgSignals(signals=sigals, fields=fields)
                        ecg_datas.append(ecg_data)
                    sample['ecg_datas'] = ecg_datas
                yield sample


class ECGQAClassificationDataset(IterableDataset):
    def __init__(self, dataset_dir: str, template_id: int, shuffle: bool = False, seed: Optional[int] = None, lazy: bool = False):
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
            "input_ecg_num": 0,
            "output_dim": 0,
            "output_classes": None,
            "question": None,

        }
        self.lazy = lazy
        self._init_classification_info()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        while self.file_index < len(self.data_files):
            if self.current_data_index >= len(self.current_file_data):
                # Load next file
                with open(self.data_files[self.file_index], 'r') as f:
                    self.current_file_data = json.load(f)
                # Shuffle samples within the file before any expensive ECG loading
                if self.shuffle and self.current_file_data and self._rng is not None:
                    self._rng.shuffle(self.current_file_data)
                self.current_data_index = 0
                self.file_index += 1
            while self.current_data_index < len(self.current_file_data):
                sample = self.current_file_data[self.current_data_index]
                if int(sample["template_id"]) != self.template_id:
                    self.current_data_index += 1
                    continue
                self.current_data_index += 1
                if not self.lazy:
                    ecg_paths = sample.get('ecg_path')
                    ecg_datas = []
                    for ecg_path in ecg_paths:
                        sigals, fields = wfdb.rdsamp(ecg_path)
                        ecg_data = EcgSignals(signals=sigals, fields=fields)
                        ecg_datas.append(ecg_data)
                    sample['ecg_datas'] = ecg_datas
                    sample["template_answer"] = sorted(
                        sample["template_answer"])

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

    def set_lazy(self, lazy: bool):
        self.lazy = lazy

    def reset_idx(self):
        self.current_data_index = 0
        self.file_index = 0


class ECGQAClassificationManager:
    def __init__(self, dataset_dir: str, class_id_path: str = "test_tools/ecgqa_info/class_id_full.csv"):
        self.class_id_df = pd.read_csv(class_id_path)
        ids_list_list = self.class_id_df["ids"].tolist()
        ids_list_list = [ast.literal_eval(id_list)
                         for id_list in ids_list_list]
        self.ids = sorted([int(item)
                          for sublist in ids_list_list for item in sublist])
        self.dataset_dict: Dict[int, ECGQAClassificationDataset] = {}
        for template_id in self.ids:
            self.create_classification_dataset(
                dataset_dir=dataset_dir, template_id=template_id)

    def create_classification_dataset(self, dataset_dir: str, template_id: int):
        self.dataset_dict[template_id] = ECGQAClassificationDataset(
            dataset_dir=dataset_dir, template_id=template_id)
