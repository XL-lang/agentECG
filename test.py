import os
import json
import numpy as np
from data.EcgQaData import EcgQaDataset
import matplotlib.pyplot as plt
from mAgents.pre_analysis_agent import pre_analysis_agent
from mAgents.data_analysis_agent import data_analysis_agent

# Initialize
ptbxl_dir_path = r"dataset/ecgqa_ptbxl/paraphrased/train"
EcgQaDataset_instance = EcgQaDataset(ptbxl_dir_path)

# Output file for results
output_file = "test_results.json"
results = []

# ECG usage documentation
usage = """
The get_lead_signals(self, lead_name: str) method retrieves the ECG signal data for a specified lead. It takes a single argument, lead_name, which must be a string matching one of the available lead names (e.g., 'II', 'V1'). The method returns a one-dimensional NumPy array of shape (n_samples,) containing the raw signal values for that lead. If the provided lead_name is not found in the signal's lead list, a ValueError is raised. For example, calling ecg.get_lead_signals('II') might return an array like [0.12, -0.03, 0.05, 0.18, -0.01, ...], representing the first few samples of lead II.

The get_lead_segment(self, lead_name: str, segment_name: str) method extracts time indices of specific ECG waveforms or intervals from a given lead. It requires two arguments: lead_name (a valid lead name) and segment_name, which must be one of the following six predefined types: 'P wave', 'QRS complex', 'T wave', 'PR interval', 'QT interval', or 'ST segment'. The method returns a list of tuples, where each tuple (start, end) denotes the sample indices of a detected segment. For instance, ecg.get_lead_segment('II', 'QRS complex') could return [(1250, 1270), (2505, 2525), ...], indicating the start and end points of detected QRS complexes in lead II.

Both methods assume the underlying ECG signal has been properly loaded and processed. The indices returned by get_lead_segment are in sample units and can be converted to time in seconds by dividing by the sampling frequency (fs). Invalid lead names or unsupported segment types will raise a ValueError. These methods are designed for programmatic access to both raw signals and structured morphological features of the ECG.
"""

# Run 50 iterations
for iteration in range(50):
    try:
        print(f"Processing iteration {iteration + 1}/50...", end=" ")
        
        # Load sample from dataset
        sample = next(iter(EcgQaDataset_instance))
        
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
        
        # Pathology Inquiry
        PI_prompt = f"Please act as an expert physician and analyze the following electrocardiogram (ECG) question. Which ECG features should be examined to answer the question? The question is {question}"
        PI_res = pre_analysis_agent.run(PI_prompt)
        
        # Data Analysis
        DA_prompt = f"""
This is the question you need to answer: {question}.

Here is the expert's analytical advice; please follow this guidance to conduct your analysis: {PI_res}.

The relevant electrocardiogram (ECG) data has already been loaded into memory, with the variable name: {ecg_names},the base info of the ecg is{info}. This is how to use the ECG class:{usage}

Use available methods and library functions as much as possible to perform the analysis, and finally provide your answer.
"""
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
        print(f"✗ Error: {str(e)}")
        continue

print(f"\nAll iterations completed. Results saved to {output_file}")
