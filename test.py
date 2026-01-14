import os
import json
import numpy as np
import matplotlib.pyplot as plt
from data_loader import initialize_dataset
from agent_runner import process_sample
from thread_executor import run_multithreaded_processing

# Initialize
ptbxl_dir_path = r"dataset/ecgqa_ptbxl/paraphrased/train"

# Output file for results
output_file = "test_results.json"

# 读取数据部分
EcgQaDataset_instance, dataset_iter = initialize_dataset(
    ptbxl_dir_path=ptbxl_dir_path,
    question_types=None,  # 使用默认的filtered_question_types
    sample_limit=1,
    shuffle=True,
    seed=42
)

# 多线程处理部分
iteration = run_multithreaded_processing(
    dataset_iter=dataset_iter,
    process_func=process_sample,
    max_workers=1,
    output_file=output_file
)

print(
    f"\nAll iterations completed after {iteration} items. Results saved to {output_file}")
