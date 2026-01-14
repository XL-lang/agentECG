import os
from data.EcgQaData import EcgQaDataset
from dataset.config import ecg_qa_types

def initialize_dataset(ptbxl_dir_path=None, question_types=None, sample_limit=1, shuffle=True, seed=42):
    """
    初始化ECG QA数据集
    
    Args:
        ptbxl_dir_path: 数据集路径，默认为 "dataset/ecgqa_ptbxl/paraphrased/train"
        question_types: 问题类型过滤列表，默认为包含"query"的类型
        sample_limit: 样本限制数量，默认为1
        shuffle: 是否打乱，默认为True
        seed: 随机种子，默认为42
    
    Returns:
        EcgQaDataset实例和迭代器
    """
    if ptbxl_dir_path is None:
        ptbxl_dir_path = r"dataset/ecgqa_ptbxl/paraphrased/train"
    
    if question_types is None:
        filtered_question_types = [i for i in ecg_qa_types if "query" in i]
    else:
        filtered_question_types = question_types
    
    EcgQaDataset_instance = EcgQaDataset(
        ptbxl_dir_path, 
        question_types=filtered_question_types, 
        sample_limit=sample_limit, 
        shuffle=shuffle, 
        seed=seed
    )
    
    dataset_iter = iter(EcgQaDataset_instance)
    
    return EcgQaDataset_instance, dataset_iter

