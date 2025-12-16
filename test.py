import os
import json
import numpy as np
from data.EcgQaData import EcgQaDataset
import matplotlib.pyplot as plt
import traceback
from mAgents.pre_analysis_agent import pre_analysis_agent
from dataset.config import get_ecgqa_answer_check_prompt,ecg_qa_types
from mAgents.data_analysis_agent import data_analysis_agent
from utils.prompt import (
    get_pathology_inquiry_prompt,
    get_ecg_analysis_prompt,
)
from mAgents.llm_checker import LLM_checker

# Initialize
ptbxl_dir_path = r"dataset/ecgqa_ptbxl/paraphrased/train"

filtered_question_types = [i for i in ecg_qa_types if "choose" in i]

EcgQaDataset_instance = EcgQaDataset(ptbxl_dir_path, question_types=filtered_question_types, sample_limit=1)

# Output file for results
output_file = "test_results.json"
results = []

dataset_iter = iter(EcgQaDataset_instance)

iteration = 0

# Loop until the dataset is fully traversed
while True:
    try:
        # Load next sample; exit when exhausted
        try:
            sample = next(dataset_iter)
        except StopIteration:
            break
        
        iteration += 1
        print(f"Processing iteration {iteration}...", end=" ")
        
        # Extract data
        question = sample["question"]
        ecgs_list = sample["ecg_datas"]
        
        # Make name mapping
        ecgs = {}
        if len(ecgs_list) == 3100:
            ecgs["ecg"] = ecgs_list[0]
        else:
            for index, ecg in enumerate(ecgs_list):
                ecgs[f"ecg_{index}"] = ecg
        
        info = ecgs_list[0].base_info()
        ecg_names = list(ecgs.keys())
        
        # Pathology Inquiry (prompt from utils)
        PI_prompt = get_pathology_inquiry_prompt(question)
        PI_res = pre_analysis_agent.run(PI_prompt)
        
        # Data Analysis (prompt from utils) + answer checker
        question_type = sample["question_type"]
        DA_prompt = get_ecg_analysis_prompt(question, question_type, PI_res, ecg_names, info)
        check_prompt = get_ecgqa_answer_check_prompt(question_type, sample["answer"],None)
        data_analysis_agent.final_answer_checks = [LLM_checker(check_prompt).check]
        data_analysis_agent.state.update(ecgs)
        final_res = data_analysis_agent.run(DA_prompt)
        
        # Prepare sample for saving (remove ecg_datas as it's not serializable)
        sample_to_save = sample.copy()
        sample_to_save.pop("ecg_datas", None)
        
        # Save results
        result_entry = {
            "iteration": iteration + 1,
            "sample": sample_to_save,
            "final_res": final_res
        }
        results.append(result_entry)
        
        # Save to file after each iteration
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("✓ Completed")
        
    except Exception as e:
        print("✗ Error encountered. Full traceback:")
        print(traceback.format_exc())
        continue

print(f"\nAll iterations completed after {iteration} items. Results saved to {output_file}")
