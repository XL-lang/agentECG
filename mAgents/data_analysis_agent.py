from smolagents import CodeAgent
from utils import code_model,clean_memory,support_oppose_check
import yaml
with open('configs/data_analysis_agent.yaml', 'r') as file:
    prompt = yaml.safe_load(file)
data_analysis_agent = CodeAgent(tools=[], 
                       model=code_model, 
                       add_base_tools=False,
                       planning_interval=10,
                       additional_authorized_imports= ["rpt.*","numpy.*","statsmodels.*","sklearn.*","scipy.*","pandas.*","ast.*","pandas.*","holidays",""],
                       description="this agent analyzes data with code after you tell it the perspective of analysis .put the data in the additional_args, x means the data, l means the label which is optional.x y are List[Any]",
                       name = "data_analysis_agent",
                       step_callbacks = [clean_memory],
                    #    final_answer_checks = [support_oppose_check],
                       prompt_templates=prompt
                       )
if __name__ == "__main__":
    data_analysis_agent.run(
        "For time series data, how to check its peaks using Python??",
        additional_args={'x':[1,2,3,20,5]}
    )

    