import traceback
from mAgents.pre_analysis_agent import create_pre_analysis_agent
from mAgents.data_analysis_agent import create_data_analysis_agent
from mAgents.llm_checker import ECGQA_EMA_checker
from utils.prompt import (
    get_pathology_inquiry_prompt,
    get_ecg_analysis_prompt,
    get_model_agent_prompt,
)
from utils.report import ECGQA_ptbxl_report


def process_sample(iter_idx, sample):
    """
    处理单个样本：运行两个agent进行分析
    
    Args:
        iter_idx: 迭代索引
        sample: 样本数据，包含question, ecg_datas, template_answer等
    
    Returns:
        tuple: (iter_idx, result_entry, error)
            - iter_idx: 迭代索引
            - result_entry: 结果字典，包含iteration, sample, final_res
            - error: 错误信息（如果有），否则为None
    """
    try:
        pre_analysis_agent = create_pre_analysis_agent()
        data_analysis_agent = create_data_analysis_agent()
        model_agent = create_pre_analysis_agent()

        question = sample["question"]
        ecgs_list = sample["ecg_datas"]

        ecgs = {}
        if len(ecgs_list) == 1:
            ecgs["ecg"] = ecgs_list[0]
        else:
            for index, ecg in enumerate(ecgs_list):
                ecgs[f"ecg_{index}"] = ecg

        
        report = ECGQA_ptbxl_report(ecgs).report()
        info = ecgs_list[0].base_info()
        ecg_names = list(ecgs.keys())
        template_answer = sample["template_answer"]
        model_agent_prompt = get_model_agent_prompt(question, ecg_names, report)  # pyright: ignore[reportArgumentType]
        model_agent_res = model_agent.run(model_agent_prompt)


        PI_prompt = get_pathology_inquiry_prompt(question, template_answer, model_agent_res)
        PI_res = pre_analysis_agent.run(PI_prompt)
        DA_prompt = get_ecg_analysis_prompt(question, template_answer, PI_res, ecg_names, info, model_agent_res)

        
        data_analysis_agent.final_answer_checks = [ECGQA_EMA_checker(template_answer).check]
        data_analysis_agent.state.update(ecgs)
        final_res = data_analysis_agent.run(DA_prompt)

        sample_to_save = sample.copy()
        sample_to_save.pop("ecg_datas", None)

        result_entry = {
            "iteration": iter_idx,
            "sample": sample_to_save,
            "final_res": final_res,
        }
        return iter_idx, result_entry, None
    except Exception:
        return iter_idx, None, traceback.format_exc()

