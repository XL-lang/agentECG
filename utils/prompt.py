from typing import Dict, List
from mAgents.fig_anaysis_tool import ECGFigAnaysiser
def get_local_models() -> str:
    name = ECGFigAnaysiser.name
    description = ECGFigAnaysiser.description
    prompt_template = f"""
The available local model is as follows:    {name}: {description}
    """
    return prompt_template
def get_pathology_inquiry_prompt(question,choices, report) -> str:
    prompt_template = f"""
    Please act as an experienced physician and analyze the following electrocardiogram (ECG) question. Which ECG features should be examined to answer the question? The question is: {question}.Please choose one or more answers from the following options: {choices}.

Use search tools to identify the relevant ECG features, then exclude those that are difficult to analyze programmatically. After curating the list, consult the search results to determine how to perform the analysis using existing packages and functions as much as possible. The ECG class you can use is {get_ecgSignals_doc_prompt()}. The available local models for ECG image analysis are: {get_local_models()}
Both ECG image analysis tools and code-based methods can be used to analyze ECG signals. Please evaluate the noise and interference robustness of the retrieved code implementations for processing each ECG feature, assign a weighted robustness score, compare them with image analysis tools, and finally select the method (either code-based or image-based) with the higher weight score.
This is the analysis result provided by the AI model. If possible, you can refer to the AI model's output to narrow down the search scope. {report}
Finally, organize your response into two parts:
1. The features that need to be analyzed, including an explanation of how each feature influences the answer.
2. Example code snippets for performing the analysis.

Do not execute the code yourself.
    """
    return prompt_template

def get_ecgSignals_doc_prompt() -> str:
    prompt_template = """
The get_lead_signals(self, lead_name: str) method retrieves the ECG signal data for a specified lead. It takes a single argument, lead_name, which must be a string matching one of the available lead names (e.g., 'II', 'V1'). The method returns a one-dimensional NumPy array of shape (n_samples,) containing the raw signal values for that lead. If the provided lead_name is not found in the signal's lead list, a ValueError is raised. For example, calling ecg.get_lead_signals('II') might return an array like [0.12, -0.03, 0.05, 0.18, -0.01, ...], representing the first few samples of lead II.

The get_lead_segment(self, lead_name: str, segment_name: str) method extracts time indices of specific ECG waveforms or intervals from a given lead. It requires two arguments: lead_name (a valid lead name) and segment_name, which must be one of the following six predefined types: 'P wave', 'QRS complex', 'T wave', 'PR interval', 'QT interval', or 'ST segment'. The method returns a list of tuples, where each tuple (start, end) denotes the sample indices of a detected segment. For instance, ecg.get_lead_segment('II', 'QRS complex') could return [(1250, 1270), (2505, 2525), ...], indicating the start and end points of detected QRS complexes in lead II.

Both methods assume the underlying ECG signal has been properly loaded and processed. The indices returned by get_lead_segment are in sample units and can be converted to time in seconds by dividing by the sampling frequency (fs). Invalid lead names or unsupported segment types will raise a ValueError. These methods are designed for programmatic access to both raw signals and structured morphological features of the ECG.
"""
    return prompt_template

def get_ecg_analysis_prompt(question, choices, PI_res, ecg_names, info, model_agent_res) -> str:
    usage = get_ecgSignals_doc_prompt()
    DA_prompt = f"""
This is the question you need to answer: {question}. Please choose one or more answers from the following options: {choices}.

Here is the expert's analytical advice; please follow this guidance to conduct your analysis: {PI_res}.

The relevant electrocardiogram (ECG) data has already been loaded into memory, with the variable name: {ecg_names},the base info of the ecg is{info}. This is how to use the ECG class:{usage}

This is the analysis result provided by the AI model. If possible, you can refer to the AI model's output to narrow down the search scope. {model_agent_res}

Use available methods and library functions as much as possible to perform the analysis, and finally provide your answer.

"""
    return DA_prompt

def get_model_agent_prompt(question:str, choices:List[str], fix_reports: Dict[str, str], ) -> str:
    gma_prompt = f"""
    Please act as a doctor to organize the conclusions of the AI model regarding ECG issues. This is the question: {question}, and these are the possible options: {choices}. This is the result of the fixed model analysis: {fix_reports}, and these are the additional models you can call upon: {""}, using the following method:.
Please organize the conclusions based on the question and the possible options, and do not retain conclusions unrelated to the question. The format should be a conclusion followed by a dict of the basis. The basis can be the confidence level of the fixed model or the analysis conclusions of the additional models.
    """
    return gma_prompt