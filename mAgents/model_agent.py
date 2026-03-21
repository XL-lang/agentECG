from smolagents import CodeAgent
from utils import code_model,clean_memory,support_oppose_check
from mAgents.fig_anaysis_tool import ECGFigAnaysiser
import yaml
with open('configs/data_analysis_agent.yaml', 'r') as file:
    prompt = yaml.safe_load(file)

def create_model_agent():
    return CodeAgent(tools=[], 
                     model=code_model, 
                     add_base_tools=False,
                     planning_interval=10,
                     additional_authorized_imports= ["rpt.*","numpy.*","statsmodels.*","sklearn.*","scipy.*","pandas.*","ast.*","pandas.*","holidays",".*","pywt.*"],
                       description="This agent invokes existing small models for answering the question.",
                       name = "modelagent",
                       step_callbacks = [clean_memory],
                    #    final_answer_checks = [support_oppose_check],
                       prompt_templates=prompt
                       )

