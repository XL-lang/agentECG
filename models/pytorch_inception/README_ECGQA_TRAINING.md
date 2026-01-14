# ECGQA分类模型训练说明

## 概述

`train_ecgqa.py` 用于训练Inception Time模型进行ECGQA分类任务。该脚本会为每个template_id训练一个独立的分类模型。

## 使用方法

### 基本用法

```bash
python train_ecgqa.py \
    --dataset_dir dataset/ecgqa_ptbxl/template/train \
    --output_dir ./output \
    --batch_size 16 \
    --epochs 15 \
    --lr 0.001 \
    --seq_len 1000
```

### 参数说明

- `--dataset_dir`: 训练数据集目录路径（默认: `dataset/ecgqa_ptbxl/template/train`）
- `--output_dir`: 输出目录，模型和检查点将保存在此目录下（默认: `./output`）
- `--batch_size`: 批次大小（默认: 16）
- `--epochs`: 训练轮数（默认: 15）
- `--lr`: 学习率（默认: 0.001）
- `--seq_len`: 序列长度（采样点数，默认: 1000）
- `--depth`: 模型深度（默认: 9）
- `--kernel_size`: 卷积核大小（默认: 60）
- `--bottleneck_size`: Bottleneck大小（默认: 32）
- `--nb_filters`: 滤波器数量（默认: 32）
- `--device`: 设备（`cuda` 或 `cpu`，默认自动检测）
- `--num_workers`: 数据加载器工作进程数（默认: 4）
- `--template_ids`: 要训练的template_id列表（逗号分隔），如果为None则训练所有（默认: None）

### 训练特定template

```bash
python train_ecgqa.py \
    --dataset_dir dataset/ecgqa_ptbxl/template/train \
    --output_dir ./output \
    --template_ids 3,56,100
```

## 输出结构

训练完成后，输出目录结构如下：

```
output/
└── templates/
    ├── template_3/
    │   ├── checkpoints/
    │   │   ├── template_3_best_checkpoint.pth
    │   │   ├── template_3_latest_checkpoint.pth
    │   │   ├── template_3_checkpoint_epoch_5.pth
    │   │   ├── template_3_checkpoint_epoch_10.pth
    │   │   └── template_3_final_model.pth
    │   └── data/
    │       ├── standard_scaler.pkl
    │       └── class_info.json
    ├── template_56/
    │   └── ...
    └── ...
```

## 模型输入输出

- **输入**: ECG信号，形状为 `(batch_size, seq_len, 12)`
  - `seq_len`: 序列长度（默认1000个采样点）
  - `12`: 12导联ECG信号
  - 如果`input_ecg_num > 1`，多个ECG信号会在时间维度拼接后下采样到`seq_len`

- **输出**: One-hot编码的分类结果，形状为 `(batch_size, output_dim)`
  - `output_dim`: 输出维度，由`classification_info['output_dim']`确定
  - 每个位置对应一个类别，值为0或1

## 数据格式

训练数据来自`ECGQAClassificationDataset`，每个样本包含：
- `ecg_datas`: List[EcgSignals]，ECG信号列表
- `answer`: List[str]，答案列表
- `template_answer`: List[str]，所有可能的答案类别（已排序）

答案会被转换为one-hot编码，基于`template_answer`中的类别顺序。

## 注意事项

1. 每个template_id会训练一个独立的模型
2. 数据集会自动分割为80%训练集和20%验证集
3. 如果某个template的数据集太小（<10个样本），会跳过训练
4. 模型使用BCELoss作为损失函数（多标签分类）
5. 评估指标包括：Loss、AUC、Accuracy、F1 Score

