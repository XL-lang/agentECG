import yaml
import os
from smolagents import LiteLLMModel
from openai import OpenAI
model_config_path = 'utils/model.yaml'
models = yaml.safe_load(open(model_config_path, 'r'))
qw_plus = models["api"]["llm"]["aliyun"]["qwen-plus"]
qwen_vl_plus = models["api"]["llm"]["aliyun"]["qwen-vl-plus"]
base_model = LiteLLMModel(model_id=qw_plus["model"], api_key=qw_plus["api_key"],api_base=qw_plus["api_base"]) 
code_model = LiteLLMModel(model_id=qw_plus["model"], api_key=qw_plus["api_key"],api_base=qw_plus["api_base"]) 
vision_language_model = LiteLLMModel(model_id=qwen_vl_plus["model"], api_key=qwen_vl_plus["api_key"],api_base=qwen_vl_plus["api_base"])
vlm_client = client = OpenAI(
    api_key=qwen_vl_plus["api_key"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
reason_client = OpenAI(
    api_key= qw_plus["api_key"],
    base_url = qw_plus["api_base"]
)
