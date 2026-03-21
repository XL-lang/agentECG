# ECGdeli ECG 分析 Skill 重构计划

## Summary

目标是基于 `external/ECGdeli` 重做一个新的 Codex skill，用于稳定执行 ECG 分析主链路，并规划其接入现有 `data_analysis_agent`。  
新 skill 不再只是“跑一次 MATLAB”，而是明确封装为面向 agent 的分析能力：输入 ECG 矩阵或文件，完成预处理、delineation、interval/amplitude 特征提取，并返回可消费的结构化结果与失败信息。

默认采用新 skill 目录，而不是继续扩展现有 `.agents/skills/ecgdeli`。现有 skill 视为原型，可保留一段过渡期，但新 agent 接入只指向新 skill。

## Key Changes

### 1. 新 skill 形态

- 新建独立 skill，建议命名为 `.agents/skills/ecgdeli-analysis`。
- `SKILL.md` 只保留触发条件、标准工作流、失败处理、何时读 reference 的说明。
- `agents/openai.yaml` 重新生成，默认 prompt 明确是“ECG 分析主链路”，而不是单纯 annotation。
- `scripts/` 下保留低自由度入口，agent 只通过包装脚本调用 MATLAB，不直接拼装 MATLAB 命令。

### 2. 对外能力与接口

- Python 入口统一暴露 3 类能力：
  - `analyze_ecg(signal, fs, ...)`：内存矩阵输入，主入口。
  - `analyze_ecg_file(input_csv, fs, ...)`：文件输入。
  - `load_ecgdeli_results(results_mat)` 或等价读取接口：把 MATLAB 输出转成 agent 友好的 Python 结构。
- 返回结构固定为一个 analysis summary，对外至少包含：
  - `status`
  - `samples`
  - `leads`
  - `beats`
  - `results_mat`
  - `summary_json`
  - `timing_features`
  - `amplitude_features`
  - `fiducial_points_available`
  - `error_identifier` / `error_message`（失败时）
- 输入约束固定为 `samples x leads`，`fs > 0`，并明确不支持在 skill 层自动猜测采样率。
- 过滤流程固定为当前 MATLAB 主链路：baseline removal -> high/low filter -> notch -> isoline correction -> annotation -> feature extraction，不在 v1 暴露可配滤波参数。

### 3. MATLAB 包装边界

- MATLAB 驱动脚本保留单一职责：读取输入、执行 ECGdeli 主链路、保存 `results.mat`、写 `summary.json`。
- 在 MATLAB 侧补齐更稳定的结构化导出，至少让 interval/amplitude 特征在 Python 侧可直接消费，而不是只依赖 `.mat` 黑盒。
- 失败时统一写出 `summary.json` 的 error 状态，保证 agent 能基于结构化错误做回退。
- 环境约束固定写入 skill：
  - 依赖本地 MATLAB
  - 依赖 `external/ECGdeli`
  - 允许通过 `ECGDELI_ROOT` 和 `matlab_bin` 覆盖默认路径

### 4. 接入 `data_analysis_agent`

- 在 `mAgents/data_analysis_agent.py` 中新增 ECGdeli skill/tool 的调用入口，不改变现有 `ECGFigAnaysiser` 的职责。
- 调整 `utils/prompt.py` 中 data-analysis prompt：
  - 将 ECGdeli 说明为“首选的结构化 ECG 分析能力”
  - 明确何时优先用 ECGdeli，何时退回 `ECG_fig_anaysis_tool`
- 接入策略固定为：
  - interval、duration、beat-level timing、波界点相关问题优先走 ECGdeli
  - 纯视觉形态、噪声敏感、难以用规则量化的问题继续路由到 `ECG_fig_anaysis_tool`
- 不把 ECGdeli 混入现有 micro-skill registry；它是外部分析能力，不是从任务中反思抽取的 micro-skill。

## Public Interfaces

- 新 skill 名称：`ecgdeli-analysis`
- 新 Python API：
  - `analyze_ecg(signal, fs, output_dir=None, matlab_bin="matlab", timeout=...)`
  - `analyze_ecg_file(input_csv, fs, output_dir=None, matlab_bin="matlab", timeout=...)`
  - `load_ecgdeli_results(results_mat)`
- `data_analysis_agent` 的可用分析能力说明将新增 ECGdeli 主链路能力描述。
- prompt 侧新增一条明确规则：凡是依赖波界点、间期、振幅特征的分析，优先调用 ECGdeli skill。

## Test Plan

- 包装层测试
  - 非法输入矩阵、空矩阵、非数值、非正 `fs` 能稳定报错。
  - 缺失 MATLAB、缺失 `ECGDELI_ROOT`、MATLAB 执行失败时返回结构化错误。
- 集成测试
  - 使用 `external/ECGdeli/Example` 样例跑通一次完整主链路。
  - 验证 `summary.json` 与 `results.mat` 均生成，且 beats/leads/samples 合理。
  - 验证 Python 侧能读取 interval/amplitude 特征，而不只是返回路径。
- Agent 接入测试
  - 对一个以 PR/QRS/QT/RR 为核心的问题，确认 `data_analysis_agent` 走 ECGdeli。
  - 对一个偏视觉 morphology 的问题，确认仍走 `ECG_fig_anaysis_tool`。
  - 对无 `fs` 或上下文不足的问题，确认 agent 不硬算 duration，而是给出未验证结论或回退。

## Assumptions

- 默认新 skill 放在 `.agents/skills/ecgdeli-analysis`，旧 `.agents/skills/ecgdeli` 暂不删除。
- v1 只覆盖分析主链路，不把 `external/ECGdeli/Filtering` 和 `ECG_Processing` 中所有 MATLAB 函数逐个公开成独立能力。
- v1 不做参数化滤波配置，不做批量任务调度，不做 GUI/可视化输出。
- `data_analysis_agent` 只规划接入 ECGdeli 的调用与路由，不改动现有 micro-skill 学习机制。
