from smolagents import CodeAgent
from utils import code_model,clean_memory,support_oppose_check
from .function_search_tool import search_tool
import yaml
with open('configs/data_analysis_agent.yaml', 'r') as file:
    prompt = yaml.safe_load(file)
def create_pre_analysis_agent():
    return CodeAgent(tools=[], 
                       model=code_model, 
                       add_base_tools=False,
                       additional_authorized_imports= ["numpy.*","statsmodels.*","sklearn.*","scipy.*","pandas.*","ast.*","pandas.*","holidays","threading"],
                       description="This agent is responsible for completing search-related tasks.",
                       name = "search_agent",
                       step_callbacks = [clean_memory],
                       prompt_templates=prompt, # type: ignore
                       )


if __name__ == "__main__":
    prompt = """
The following will present a question and options. You need to extract the time-series operations involved in the question and options, such as STL decomposition, stability analysis, trend analysis, etc. Then, use the function_search_tool to find methods for performing these operations. Finally, summarize the results into a dictionary {"operation name": str(content and precautions)}.
question and options:
Here are two sequences, both showing the trend of changes in the number of tourists over a certain year.  
A. Both curves have two peaks.  
B. Both curves decline on June 27th.
"""
    create_pre_analysis_agent().run(
        prompt,
    )