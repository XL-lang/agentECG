from smolagents import CodeAgent
from utils import code_model,clean_memory,support_oppose_check
from mAgents.fig_anaysis_tool import ECGFigAnaysiser
from mAgents.ecgdeli_analysis_tool import ECGDeliMeasurementTool, ECGDeliPrepareTool
import yaml
with open('configs/data_analysis_agent.yaml', 'r') as file:
    prompt = yaml.safe_load(file)

def create_data_analysis_agent():
    return CodeAgent(tools=[ECGFigAnaysiser(), ECGDeliPrepareTool(), ECGDeliMeasurementTool()], 
                     model=code_model, 
                     add_base_tools=False,
                     planning_interval=10,
                     additional_authorized_imports= ["rpt.*","numpy.*","statsmodels.*","sklearn.*","scipy.*","pandas.*","ast.*","pandas.*","holidays",".*","pywt.*"],
                       description="This agent analyzes ECG data with reusable micro-skills learned from prior data_analysis_agent tasks. It should prefer ECGDeli_prepare_tool plus ECGDeli_measurement_tool for fiducial-point, interval, duration, wave-slice, segment-boundary, and amplitude questions when fs is known, keep ECG_fig_anaysis_tool as a morphology or visual fallback rather than a primary slicing tool, and prefer the visual tool first for noise or artifact classification. ECGDeli_prepare_tool returns a session summary payload containing `session_id`; extract that field once and pass only the raw session_id string to ECGDeli_measurement_tool.",
                       name = "data_analysis_agent",
                       step_callbacks = [clean_memory],
                    #    final_answer_checks = [support_oppose_check],
                       prompt_templates=prompt
                       )
if __name__ == "__main__":
    create_data_analysis_agent().run(
        "For time series data, how to check its peaks using Python??",
        additional_args={'x':[1,2,3,20,5]}
    )

    
