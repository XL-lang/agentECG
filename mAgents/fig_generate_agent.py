from smolagents import CodeAgent
from utils import code_model,clean_memory
from .data_analysis_agent import data_analysis_agent




fig_generate_agent = CodeAgent(tools=[], 
                        planning_interval=4,
                       model=code_model, 
                       managed_agents=[data_analysis_agent],
                       description="this agent draws the figure using matplotlib",
                       add_base_tools=True,
                       additional_authorized_imports= ["numpy.*","Statsmodels.*","scikit-learn.*","scipy.*","pandas.*","matplotlib.*","seaborn.*","plotly*.","bokeh.*","ast.*"],
                       name = "fig_generate_agent",
                       step_callbacks= [clean_memory],
                       
                       )
if __name__ == "__main__":
    fig_generate_agent.run(
        "can you draw a figure for the data? save the figure to local",
        additional_args={'x':[1,2,3,4,5,6,7,8,9,10]}
    )
