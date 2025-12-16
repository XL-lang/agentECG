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
from .EcgSignals import EcgSignals
from dataset.config import ecg_qa_types

class EcgQaDataset(IterableDataset):
    def __init__(self,dataset_dir:str,question_types:Optional[List[str]]=None,sample_limit:Optional[int]=None):
        """
        PyTorch IterableDataset for ECG-QA data.

        Args:
            dataset_dir (str): Path to the ECG-QA dataset directory.
        """
        super(EcgQaDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.data_files = glob.glob(os.path.join(self.dataset_dir, "*.json"))
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