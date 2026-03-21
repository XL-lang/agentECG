from typing import Dict, List
from models.pytorch_inception.ecgqa_classifier import ECGQAClassifierManager
from mAgents.fig_anaysis_tool import ECGFigAnaysiser
from mAgents.ecgdeli_analysis_tool import ECGDeliMeasurementTool, ECGDeliPrepareTool


def get_local_models() -> str:
    tool_descriptions = [
        f"{ECGDeliPrepareTool.name}: {ECGDeliPrepareTool.description}",
        f"{ECGDeliMeasurementTool.name}: {ECGDeliMeasurementTool.description}",
        f"{ECGFigAnaysiser.name}: {ECGFigAnaysiser.description}",
    ]
    prompt_template = f"""
The available local analysis tools are as follows:
    {tool_descriptions[0]}
    {tool_descriptions[1]}
    {tool_descriptions[2]}
    """
    return prompt_template
def get_pathology_inquiry_prompt(
    question,
    choices,
    report,
    skill_context: str | None = None,
    local_pre_analysis_context: str | None = None,
    remote_search_needed: bool = True,
    open_questions_for_remote_search: list[str] | None = None,
) -> str:
    reusable_skill_context = get_skill_context_prompt(skill_context)
    has_reusable_skills = skill_context is not None and "None retrieved from prior tasks" not in skill_context
    local_context = local_pre_analysis_context or "No local pre-analysis context was prepared."
    open_questions = open_questions_for_remote_search or []
    remote_questions_text = "\n".join(f"- {item}" for item in open_questions) if open_questions else "- No remote search questions are currently open."

    if not remote_search_needed:
        search_policy = f"""
Local-first pre-analysis is sufficient for this task. Do not call remote search tools.

Local pre-analysis context:
{local_context}

Reusable data-analysis micro-skills:
{reusable_skill_context}
"""
        output_requirements = """
Return a concise local analysis plan with the following sections:
1. target_features
2. local_tool_plan
3. evidence_priority
4. known_risks
5. final_search_needed: no
"""
    elif has_reusable_skills:
        search_policy = f"""
Local-first pre-analysis must be performed before any remote search.

Local pre-analysis context:
{local_context}

Reusable data-analysis micro-skills:
{reusable_skill_context}

Only use remote search as a supplement for the following open medical knowledge gaps:
{remote_questions_text}

Because reusable skills already cover code/tool operation patterns, do not search for or include Python code snippets, package calls, implementation recipes, tool substitutions, or API examples.
If remote search is used, limit it to ECG feature definitions, normal ranges, thresholds, diagnostic criteria, lead-specific interpretation rules, and clinical explanations.
"""
        output_requirements = """
Return a structured analysis plan with the following sections:
1. target_features
2. local_tool_plan
3. remote_medical_knowledge
4. evidence_priority
5. known_risks
6. final_search_needed: yes or no
Do not include code snippets.
"""
    else:
        search_policy = f"""
Local-first pre-analysis must be performed before any remote search.

Local pre-analysis context:
{local_context}

The ECG class you can use is {get_ecgSignals_doc_prompt()}
The available local models for ECG image analysis are: {get_local_models()}

Remote search is allowed only to supplement the following missing items:
{remote_questions_text}

Prefer local ECG APIs and local tools when deciding how a feature should be analyzed. Do not let remote search dominate tool selection.
If remote search is used, focus on ECG feature definitions, clinical thresholds, diagnostic criteria, lead-specific interpretation rules, and how each feature influences the answer.
"""
        output_requirements = """
Return a structured analysis plan with the following sections:
1. target_features
2. local_tool_plan
3. remote_medical_knowledge
4. evidence_priority
5. known_risks
6. final_search_needed: yes or no
Do not include code snippets.
"""

    prompt_template = f"""
    Please act as an experienced physician and analyze the following electrocardiogram (ECG) question. Which ECG features should be examined to answer the question? The question is: {question}.Please choose one or more answers from the following options: {choices}.

{search_policy}
This is the analysis result provided by the AI model. If possible, you can refer to the AI model's output to narrow down the search scope. {report}
{output_requirements}

Pre-analysis is a planning stage, not an execution stage.
Do not call `ECG_fig_anaysis_tool`, `ECGDeli_prepare_tool`, or `ECGDeli_measurement_tool` in this stage.
Only use remote search when the local plan explicitly leaves open medical knowledge gaps.
Do not execute the code yourself.
    """
    return prompt_template

def get_ecgSignals_doc_prompt() -> str:
    prompt_template = f"""
The get_lead_signals(self, lead_name: str) method retrieves the ECG signal data for a specified lead. It takes a single argument, lead_name, which must be a string matching one of the available lead names (e.g., 'II', 'V1'). The method returns a one-dimensional NumPy array of shape (n_samples,) containing the raw signal values for that lead. If the provided lead_name is not found in the signal's lead list, a ValueError is raised. For example, calling ecg.get_lead_signals('II') might return an array like [0.12, -0.03, 0.05, 0.18, -0.01, ...], representing the first few samples of lead II.

The get_lead_segment(self, lead_name: str, segment_name: str) method extracts time indices of specific ECG waveforms or intervals from a given lead. It requires two arguments: lead_name (a valid lead name) and segment_name, which must be one of the following six predefined types: 'P wave', 'QRS complex', 'T wave', 'PR interval', 'QT interval', or 'ST segment'. The method returns a list of tuples, where each tuple (start, end) denotes the sample indices of a detected segment. For instance, ecg.get_lead_segment('II', 'QRS complex') could return [(1250, 1270), (2505, 2525), ...], indicating the start and end points of detected QRS complexes in lead II.

Both methods assume the underlying ECG signal has been properly loaded and processed. The indices returned by get_lead_segment are in sample units and can be converted to time in seconds by dividing by the sampling frequency (fs). Invalid lead names or unsupported segment types will raise a ValueError. These methods are designed for programmatic access to both raw signals and structured morphological features of the ECG.
"""
    return prompt_template
def get_skill_context_prompt(skill_context: str | None) -> str:
    if skill_context is None or not skill_context.strip():
        return "No reusable micro-skills were retrieved from prior data_analysis_agent tasks."
    return skill_context

def get_ecg_analysis_prompt(
    question,
    choices,
    PI_res,
    ecg_names,
    model_agent_res,
    skill_context: str | None = None,
    routing_context: str | None = None,
) -> str:
    usage = get_ecgSignals_doc_prompt()
    reusable_skill_context = get_skill_context_prompt(skill_context)
    routing_prompt = routing_context or "ECG tool routing verdict:\n- preferred_tool: none"
    DA_prompt = f"""
This is the question you need to answer: {question}. Please choose one or more answers from the following options: {choices}.

Here is the expert's analytical advice; please follow this guidance to conduct your analysis: {PI_res}.

The relevant electrocardiogram (ECG) data has already been loaded into memory, with the variable name: {ecg_names}. This is how to use the ECG class:{usage}

This routing verdict is task-specific and has higher priority than the reusable micro-skills below:
{routing_prompt}

The following reusable micro-skills were retrieved from prior successful `data_analysis_agent` tasks. They are capabilities, not facts about the current ECG. Reuse them when they fit the present task, and ignore them when they do not:
{reusable_skill_context}

This is the analysis result provided by the AI model. If possible, you can refer to the AI model's output to narrow down the search scope. {model_agent_res}

Available local analysis tool:
{get_local_models()}

Execution requirements:
- Follow the routing verdict first; treat it as task-specific guidance for tool selection.
- If `preferred_tool` is `ECGDeli_measurement_tool`, call `ECGDeli_prepare_tool` first and reuse its `session_id` for the smallest ECGDeli measurement needed before visual inspection unless prerequisites fail.
- `ECGDeli_prepare_tool` returns a JSON-like session summary payload with a `session_id` field. Extract that field once and store the raw string, then pass only that raw string into `ECGDeli_measurement_tool`.
- Do not pass the full prepare-tool payload as `session_id`, and do not write code that indexes an already extracted session_id string again.
- If `preferred_tool` is `ECG_fig_anaysis_tool`, use it first only because the task is morphology-heavy or visually defined.
- If `requires_fs` is true and `fs_available` is false, do not produce duration-based conclusions from hard-coded assumptions.
- Use the fallback tool only if the preferred path fails or cannot satisfy its prerequisites.
- Use the retrieved micro-skills as reusable methods or routing hints, not as evidence about the current ECG.
- Prefer minimal, verifiable code and reuse existing ECG APIs whenever possible.
- Do not invent undocumented ECG properties, helper methods, or external metadata.
- If duration-style conclusions depend on `fs`, verify that requirement instead of hard-coding assumptions.
- Prefer `ECGDeli_measurement_tool` for fiducial-point, wave-boundary, interval, duration, beat-level timing, amplitude-feature, and per-wave slicing questions when `fs` is available.
- If `fs` is missing or unreliable, do not hard-code durations; either state the limitation or use a non-duration fallback.
- For ECGdeli, do not ask for a full summary. Prepare one session, then query only the specific beat/lead/wave/feature needed for the claim.
- Treat ECG segmentation, wave slicing, and boundary extraction as ECGdeli responsibilities when ECGdeli prerequisites are satisfied.
- ECGdeli wave slices are summarized by default. Use `include_samples=True` only when the full waveform values are strictly necessary.
- Route morphology-heavy or visually defined claims to `ECG_fig_anaysis_tool` when simple numeric logic is fragile.
- Route noise, artifact, or interference classification to `ECG_fig_anaysis_tool` before attempting ECGdeli measurements.
- Do not use `ECG_fig_anaysis_tool` as the primary slicing tool when ECGdeli can provide the needed segment boundaries or wave slice.
- Final answers must be restricted to the provided options and supported by evidence from this task.

Please provide:
1. The features that need to be analyzed, including an explanation of how each feature influences the answer.
2. Example code snippets or tool usage for the present task.
3. A final answer grounded in the validated evidence from this task.

"""
    return DA_prompt

def get_model_agent_prompt(question:str, choices:List[str], model_prompt:str, model_select_res:str ) -> str:
    gma_prompt = f"""
    Please act as a professional ECG analyst. Here is the question you need to analyze: {question}, here are the possible answers related to the question: {choices}, and here is the description and calling method for the models you can invoke: {model_prompt}. Please call the model based on the question and finally return the analysis results. Note that you should select the most suitable model as much as possible and avoid using too many models to prevent interference with the results. 
    The expert's advice on model selection is: {model_select_res}.
    The analysis results should contain the following two parts. 
    1. Conclusion. 
    2. Which model was used (the function of the model needs to be explained) to obtain what results, and how these results led to the final conclusion.
    """
    return gma_prompt
def get_model_agent_check_prompt(question:str, choices:List[str] ) -> str:
    gma_check_prompt = f"""
    This is question {question}, and you are to act as a reviewer to assess whether the response to the question meets the specified requirements. Please conduct the review according to the following criteria:

Responses to the question fall into two categories:

1. The model provides a direct answer to the question:  
   - The response must explicitly state that it is a direct answer to the question.  
   - The content of the answer must come exclusively from one or more options provided in {choices}.

2. No direct answer is available:  
   - The response must clearly state that the model cannot directly answer the question.  
   - However, it may provide relevant reference information, which must be practically helpful.

Additionally, the final response must include the following elements:  
- The name of the model used and a description of its functionality.  
- The specific result obtained after execution.  
- An explanation of how this result was used to form either a direct answer or reference information.  
- If it is a direct answer, verification must be included confirming that the model's functionality genuinely supports providing a direct answer to this type of question.

Below is a checklist based on the above rules:

1. Is there a clear distinction between the two scenarios: “direct answer available” and “no direct answer”?  
2. If it is a direct answer, does the response explicitly declare it as such?  
3. If it is a direct answer, is its content strictly limited to the options provided in {choices}?  
4. If there is no direct answer, does the response clearly state that the model cannot directly answer the question?  
5. If there is no direct answer, does it provide useful and relevant reference information?  
6. Does the response specify the name of the model used and describe its functionality?  
7. Does it describe the specific result obtained after execution?  
8. Does it explain how this result was used to generate either a direct answer or reference information?  
9. If it is a direct answer, does it verify that the model's functionality indeed supports directly answering this type of question?

Please check each item against the above checklist. If all requirements are met, append #PASS at the end of your review. If any requirement is not met, list each unmet criterion with a clear explanation of why it fails.
Here is the answer: 
"""
    return gma_check_prompt
def get_ecgqaclassifier_usage_prompt(eqcm:ECGQAClassifierManager) -> str:
    eqcm_usage_prompt = f"""
    The following dict represents the model's ID (int) and the question the model answered: {eqcm.question}, below is a usage example

    ```python

    report = eqcm.report_result(666) # 666 represents the model ID. 666 is not a real ID; please select an appropriate one.
    print(f"report from model {666} is ", report) # The ECG data has been pre-loaded into the EQCM instance. Therefore, calling this method allows for direct analysis without the need to pass the ECG data again.

    ```
    """
    return eqcm_usage_prompt

def get_final_result_prompt(question:str, choices:List[str], model_res:str, medical_res:str, code_res:str) -> str:
    final_result_prompt = f"""
    You need to answer this question {question}, and the answer is one or more of the following {choices}. This is the conclusion from the AI model: {model_res},  
This is the relevant medical content obtained from the search: {medical_res},  
This is the conclusion from the code analysis: {code_res}.  
If the AI model provides a direct answer to the question, please combine the medical content to answer the question and include the basis for the analysis (if the code analysis provides supporting evidence, you may incorporate the conclusion from the code analysis).  
If the AI model does not provide a direct answer to the question, please use the conclusion from the code analysis and include the basis for the analysis.
    """
    return final_result_prompt
def select_models_prompt(question:str, choices:str) -> str:
    select_models_prompt = f"""
    This is the question I need to answer{question}, and this is the list of models capable of automatic processing{choices}. 
    Please select one or more models based on the question (represented by model ID), so that I can process the question using the models you selected. 
    Choose the model IDs that are the closest match. The answer should be the model ID.
    """
    return select_models_prompt
def get_final_result_checker_prompt(question:str, choices:List[str], model_res:str, medical_res:str, code_res:str, ) -> str:
    final_result_checker_prompt = f"""
    This is question {question}. Please act as a reviewer to evaluate whether the final answer meets the requirements. Conduct your review according to the following criteria:

The final answer must follow different processing logic depending on whether the AI model provides a direct response to the question:

If the AI model provides a direct answer to the question:  
- Combine the AI model's response with medical knowledge to formulate the answer.  
- Analysis rationale must be included. If the code analysis conclusion supports the answer, it should be prioritized for citation or integration, but the code analysis conclusion must not override or interfere with the AI model's answer.

If the AI model does not provide a direct answer to the question:  
- The code analysis conclusion should serve as the primary basis, accompanied by a reasonable analytical process.

Additionally, regardless of the scenario, the final answer must satisfy all the following requirements:  
- Any cited answer(s) must be strictly limited to one or more options from {choices} (i.e., the provided list of answer choices).  
- The source of information used (AI model conclusion / medical content / code analysis conclusion) must be explicitly stated.  
- If code analysis conclusions are used, explain how they support the final judgment.  
- If medical content is referenced, explain how it corroborates the answer.

Below is the checklist based on the above rules:  
- Was the determination of “whether the AI model provided a direct answer” correctly made?  
- If there was a direct answer, was supporting analysis provided (preferably supported by code analysis)?  
- If there was a direct answer, when deriving the final answer, were only the model's response and medical knowledge used, with code analysis serving solely as supporting evidence rather than the basis for judgment?  
- If there was no direct answer, was the code analysis conclusion adopted and its basis explained?  
- Are the selected answers strictly confined to the options provided in {choices}?  
- Are the sources of all information clearly indicated (AI model / medical content / code analysis)?  
- Is the overall logic coherent, and is the conclusion reasonable and well-supported?

Please verify each item against this checklist. If all requirements are met, append "#PASS" at the end of your review. If any requirement is not met, list each point of non-compliance with specific reasons.


Here is the model analysis conclusion: {model_res}  
Here is the medical knowledge: {medical_res}  
Here is the code analysis conclusion: {code_res}
Here is the final answer: 
    """
    return final_result_checker_prompt
