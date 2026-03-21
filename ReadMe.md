# agentECG

面向 ECG 问答与技能构建的多 Agent 项目。仓库默认不提交真实模型密钥和数据集内容；本地运行前需要先准备模型配置、可选知识库连接和数据集目录。

## 1. 环境配置

### 1.1 Python / Conda

建议使用独立环境运行：

```bash
conda create -n agentecg python=3.10 -y
conda activate agentecg
```

项目根目录没有统一的依赖锁文件，至少需要保证下列核心依赖可用：

- `smolagents`
- `openai`
- `PyYAML`
- `qdrant-client`
- 项目运行所需的数值与 ECG 处理依赖

如果你还要训练 `models/pytorch_inception` 下的分类模型，另见 [models/pytorch_inception/README_ECGQA_TRAINING.md](/home/xl/agentECG/models/pytorch_inception/README_ECGQA_TRAINING.md)。

### 1.2 模型密钥配置

真实密钥文件使用本地私有配置 [utils/model.yaml](/home/xl/agentECG/utils/model.yaml)，该文件已加入 `.gitignore`，不会默认提交。

先从模板复制：

```bash
cp utils/model.example.yaml utils/model.yaml
```

然后填写你自己的密钥。当前代码 [utils/model.py](/home/xl/agentECG/utils/model.py) 会直接读取这个 YAML，至少需要配置：

- `api.llm.aliyun.qwen-plus`
- `api.llm.aliyun.qwen-vl-plus`
- `api.search-engine.google`

说明：

- `qwen-plus`：文本推理和搜索问答使用。
- `qwen-vl-plus`：VLM 切片审计等视觉能力使用。
- `google.api_key`：保留在配置模板中；是否实际使用取决于你的上层搜索链路。

### 1.3 知识库连接

知识库检索由 [mAgents/function_search_tool.py](/home/xl/agentECG/mAgents/function_search_tool.py) 读取环境变量配置，不走 `utils/model.yaml`。

支持的环境变量：

- `SEARCH_KB_BACKEND`
- `QDRANT_URL`
- `QDRANT_HOST`
- `QDRANT_PORT`
- `QDRANT_COLLECTION`
- `QDRANT_API_KEY`
- `QDRANT_TOP_K`
- `QDRANT_QUERY_MODEL`
- `SEARCH_KB_TIMEOUT_SECONDS`

最小可用示例：

```bash
export SEARCH_KB_BACKEND=qdrant
export QDRANT_URL=http://127.0.0.1:6333
export QDRANT_COLLECTION=ecg
export QDRANT_TOP_K=5
export QDRANT_QUERY_MODEL=sentence-transformers/all-MiniLM-L6-v2
export SEARCH_KB_TIMEOUT_SECONDS=2
```

可选替代写法：

- 不写 `QDRANT_URL` 时，可以改用 `QDRANT_HOST` + `QDRANT_PORT`。
- 私有部署时可额外设置 `QDRANT_API_KEY`。

如果没有配置 Qdrant，或 Qdrant 健康检查 / 查询失败，`search_tool` 会自动回退到联网搜索。

## 2. 数据集位置

### 2.1 默认目录

[data_loader.py](/home/xl/agentECG/data_loader.py) 默认读取：

```text
dataset/ecgqa_ptbxl/paraphrased/train
```

也就是说，运行主流程前至少要让 ECG-QA 派生数据出现在这个目录，或者在代码里显式传入 `ptbxl_dir_path`。

### 2.2 下载方式

仓库根目录的 [load_dataset.sh](/home/xl/agentECG/load_dataset.sh) 会把数据下载到 `dataset/` 下，包含：

- `ecg-qa` Git 仓库
- PTB-XL 压缩包下载与解压

运行示例：

```bash
bash load_dataset.sh
```

说明：

- `dataset/` 目录已保留在仓库中，但真实数据内容默认不进入 Git。
- 数据目录较大，不建议直接提交到 GitHub。
- 如果你已经手动准备好了数据，只要路径和代码约定一致，不必执行下载脚本。

## 3. 使用方法

### 3.1 最小运行示例

主流程由三部分组成：

- [data_loader.py](/home/xl/agentECG/data_loader.py) 负责构造数据集迭代器
- [thread_executor.py](/home/xl/agentECG/thread_executor.py) 负责并发调度和结果落盘
- [agent_runner.py](/home/xl/agentECG/agent_runner.py) 中的 `process_sample(...)` 负责单样本 Agent 流水线

最小示例：

```python
from data_loader import initialize_dataset
from thread_executor import run_multithreaded_processing
from agent_runner import process_sample

_, dataset_iter = initialize_dataset(
    ptbxl_dir_path="dataset/ecgqa_ptbxl/paraphrased/train",
    question_types=["single-choose"],
    sample_limit=20,
    shuffle=True,
    seed=42,
)

run_multithreaded_processing(
    dataset_iter=dataset_iter,
    process_func=process_sample,
    max_workers=2,
    output_file="tmp/eval_results.json",
    save_agent_mem=False,
    agent_mem_file="tmp/eval_agent_mem.json",
    run_mode="eval",
)
```

### 3.2 关键参数

`run_multithreaded_processing(...)` 当前关键参数如下：

- `max_workers`：并发 worker 数
- `output_file`：结果 JSON 输出路径
- `save_agent_mem`：是否把 agent memory 落盘
- `agent_mem_file`：agent memory 文件路径
- `run_mode`：运行模式，当前支持 `eval` 和 `skill_build`

建议：

- 普通评测建议 `max_workers=2~5`
- `skill_build` 模式建议并发更低，避免反思与检索链路同时放大 API 压力

### 3.3 `eval` 模式

`eval` 用于常规跑分和结果验证，不做技能生成落库。

推荐配置：

```python
run_multithreaded_processing(
    dataset_iter=dataset_iter,
    process_func=process_sample,
    max_workers=2,
    output_file="tmp/run_eval.json",
    save_agent_mem=False,
    agent_mem_file="tmp/run_eval_agent_mem.json",
    run_mode="eval",
)
```

特点：

- 不从 registry 读取可复用技能
- 不执行失败反思后的 skill build 逻辑
- 输出中仍会记录 `run_mode`

### 3.4 `skill_build` 模式

`skill_build` 用于构建和更新可复用微技能。

推荐配置：

```python
run_multithreaded_processing(
    dataset_iter=dataset_iter,
    process_func=process_sample,
    max_workers=1,
    output_file="tmp/run_skill_build.json",
    save_agent_mem=False,
    agent_mem_file="tmp/run_skill_build_agent_mem.json",
    run_mode="skill_build",
)
```

特点：

- 会在 pre-analysis 和 data-analysis 阶段检索已有技能
- 答错时会触发 VLM 切片审计、反思与 retry
- 成功样本会尝试抽取可复用技能并写入 registry

默认技能注册表位于：

```text
agent_reflect/storage/agent_skill_registry.json
```

### 3.5 后台运行

示例：

```bash
conda run -n agentecg python your_run_script.py
```

或：

```bash
nohup conda run -n agentecg python your_run_script.py > run.log 2>&1 &
```

如果你沿用仓库里的临时脚本风格，也可以把单次实验脚本放在 `tmp/` 下，再用上述方式启动。

## 4. 额外说明

- `utils/model.yaml` 是本地私有文件，不要提交。
- 如果真实密钥曾经进入 Git 历史，应立即去服务端轮换。
- 分类模型训练说明不放在主 README，单独见 [models/pytorch_inception/README_ECGQA_TRAINING.md](/home/xl/agentECG/models/pytorch_inception/README_ECGQA_TRAINING.md)。
