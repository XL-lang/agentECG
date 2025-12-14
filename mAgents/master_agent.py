from smolagents import CodeAgent
import yaml
from utils import code_model,abcd_check
from .data_analysis_agent import data_analysis_agent
with open('configs/master_agent.yaml', 'r') as file:
    prompt_templates = yaml.safe_load(file)
master_agent = CodeAgent(tools=[], 
                       model=code_model, 
                       description="this the main agent of the project,it plans and solves the problem step by step with other agents ",
                       name = "main_agent",
                       prompt_templates=prompt_templates,
                       final_answer_checks= [abcd_check],
                       )
if __name__ == "__main__":
    master_agent.run(
        "can you tell me if the data is stable?。use the data_analysis_agent to analyze the data",
        additional_args={'x':[1,2,3,4,5,6,7,8,9,10]}
    )