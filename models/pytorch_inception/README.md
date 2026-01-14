# Inception Time模型 - PyTorch实现

这是Inception Time模型在PTB-XL数据集上的PyTorch实现，用于ECG信号的71分类（all任务）任务。

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (可选，用于GPU加速)

## 安装依赖

```bash
# 安装PyTorch（根据你的CUDA版本选择）
# CPU版本
pip install torch torchvision torchaudio

# CUDA 11.8版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

或者使用conda：
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 数据集

数据集路径：`/home/xl/dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/`

确保数据集包含以下文件：
- `ptbxl_database.csv`
- `scp_statements.csv`
- `records100/` 或 `records500/` 目录

## 训练模型

### 基本训练（71分类 - all任务）

```bash
python train.py \
    --datafolder /home/xl/dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ \
    --output_dir ./output \
    --task all \
    --batch_size 16 \
    --epochs 15 \
    --lr 0.001 \
    --depth 9 \
    --kernel_size 60
```

### 训练参数说明

- `--datafolder`: 数据集文件夹路径
- `--output_dir`: 输出目录（保存checkpoints和预处理数据）
- `--task`: 任务类型（all, diagnostic, subdiagnostic, superdiagnostic, form, rhythm）
- `--sampling_rate`: 采样率（100或500，默认100）
- `--batch_size`: 批次大小（默认16）
- `--epochs`: 训练轮数（默认15）
- `--lr`: 学习率（默认0.001）
- `--depth`: 模型深度（默认9）
- `--kernel_size`: 卷积核大小（默认60）
- `--bottleneck_size`: Bottleneck层大小（默认32）
- `--nb_filters`: 滤波器数量（默认32）
- `--device`: 设备（cuda/cpu，默认自动检测）

### 训练配置（对应README中的配置）

对于"all"任务（71分类），使用以下配置：
- epochs: 15
- batch_size: 16
- learning_rate: 0.001
- model_depth: 9
- kernel_size: 60
- loss: BCE (Binary Cross-Entropy)

## 测试模型

```bash
python test.py \
    --checkpoint ./output/checkpoints/best_checkpoint.pth \
    --datafolder /home/xl/dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ \
    --output_dir ./output \
    --task all \
    --save_predictions
```

## 输出文件

训练完成后，文件保存在 `output/` 目录：

```
output/
├── data/
│   ├── mlb.pkl                    # MultiLabelBinarizer
│   └── standard_scaler.pkl        # StandardScaler
├── checkpoints/
│   ├── best_checkpoint.pth        # 最佳模型（基于验证集AUC）
│   ├── latest_checkpoint.pth      # 最新checkpoint
│   ├── checkpoint_epoch_5.pth     # 每5个epoch的checkpoint
│   ├── checkpoint_epoch_10.pth
│   ├── checkpoint_epoch_15.pth
│   └── final_model.pth            # 最终模型
└── predictions/                   # 测试预测结果（如果使用--save_predictions）
    ├── y_test_pred.npy
    └── y_test_true.npy
```

## 模型架构

Inception Time模型基于Inception架构，包含：
- 多个Inception模块，每个模块使用不同大小的卷积核
- 残差连接（每3个模块）
- Global Average Pooling
- 全连接层 + Sigmoid激活（多标签分类）

## 性能指标

根据原始论文，Inception Time模型在PTB-XL数据集"all"任务上的预期性能：
- Macro AUC: ~0.926

## 注意事项

1. **GPU内存**: 如果遇到GPU内存不足，可以减小batch_size
2. **数据缓存**: 首次加载数据时会创建缓存文件（raw100.npy或raw500.npy），后续训练会直接使用缓存
3. **多进程**: 如果遇到数据加载问题，可以设置 `--num_workers 0`

## 故障排除

### CUDA内存不足
```bash
# 减小batch_size
python train.py --batch_size 8 ...
```

### 数据加载错误
```bash
# 使用单进程加载
python train.py --num_workers 0 ...
```

## 使用训练好的模型进行推理

### InceptionClassifier类

使用 `InceptionClassifier` 类可以方便地加载训练好的模型并进行推理：

```python
from classifier import InceptionClassifier
import numpy as np

# 加载分类器
classifier = InceptionClassifier(
    checkpoint_path='./output/checkpoints/best_checkpoint.pth',
    data_dir='./output/data'  # 包含scaler和mlb的目录
)

# 准备ECG信号（形状为 (:, 12) 或 (N, :, 12)）
ecg_signal = np.random.randn(1000, 12).astype(np.float32)  # 示例：1000个时间点，12个导联

# 预测（只返回置信度>0.5的类别）
result = classifier.predict(ecg_signal)
# 返回: {"class_name": confidence, ...}

# 获取所有类别的概率
all_probs = classifier.predict_proba(ecg_signal)
# 返回: {"class_name": confidence, ...} (所有类别)

# 批量预测（多个样本）
ecg_signals = np.random.randn(5, 1000, 12).astype(np.float32)  # 5个样本
results = classifier.predict(ecg_signals)
# 返回: [{"class_name": confidence, ...}, ...] (列表)
```

### 输入输出说明

- **输入**: ECG信号，numpy array
  - 单个样本: 形状为 `(:, 12)`，例如 `(1000, 12)`
  - 批量样本: 形状为 `(N, :, 12)`，例如 `(5, 1000, 12)`
  - 如果长度不是1000，会自动裁剪或填充

- **输出**: 字典或字典列表
  - 单个样本: `{"class_name": confidence, ...}`，confidence是float类型（0-1之间）
  - 批量样本: `[{"class_name": confidence, ...}, ...]`
  - 默认只返回置信度>0.5的类别，使用 `return_all=True` 可返回所有类别

### 命令行使用

```bash
python classifier.py ./output/checkpoints/best_checkpoint.pth ./output/data
```

## 引用

如果使用此代码，请引用原始论文：
> N. Strodthoff, P. Wagner, T. Schaeffter, and W. Samek, 'Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL', IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 5, pp. 1519–1528, May 2021.

