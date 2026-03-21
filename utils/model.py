import os
from importlib.util import find_spec

import yaml


def _disable_socks_proxy_if_unsupported():
    # Keep HTTP(S) proxy support, but drop SOCKS proxy env vars when socksio
    # is unavailable so httpx/litellm can initialize successfully.
    all_proxy = os.environ.get("ALL_PROXY") or os.environ.get("all_proxy")
    if not all_proxy:
        return
    if not all_proxy.lower().startswith("socks"):
        return
    if find_spec("socksio") is not None:
        return
    os.environ.pop("ALL_PROXY", None)
    os.environ.pop("all_proxy", None)


_disable_socks_proxy_if_unsupported()

from smolagents import LiteLLMModel
from openai import OpenAI
model_config_path = 'utils/model.yaml'
models = yaml.safe_load(open(model_config_path, 'r'))
qw_plus = models["api"]["llm"]["aliyun"]["qwen-plus"]
qwen_vl_plus = models["api"]["llm"]["aliyun"]["qwen-vl-plus"]
base_model = LiteLLMModel(
    model_id=qw_plus["model"], api_key=qw_plus["api_key"], api_base=qw_plus["api_base"])
code_model = LiteLLMModel(
    model_id=qw_plus["model"], api_key=qw_plus["api_key"], api_base=qw_plus["api_base"])
vision_language_model = LiteLLMModel(
    model_id=qwen_vl_plus["model"], api_key=qwen_vl_plus["api_key"], api_base=qwen_vl_plus["api_base"])
vlm_client = OpenAI(
    api_key=qwen_vl_plus["api_key"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
reason_client = OpenAI(
    api_key=qw_plus["api_key"],
    base_url=qw_plus["api_base"]
)
