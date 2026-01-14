"""
训练Inception Time模型用于ECGQA分类任务
为每个template_id训练一个独立的模型
"""
import wfdb
from .inception_time import create_inception_time_model
from data.EcgSignals import EcgSignals
from data.EcgQaData import ECGQAClassificationManager
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import json

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


class ECGQAClassificationDatasetWrapper(Dataset):
    """将ECGQAClassificationDataset包装为PyTorch Dataset（懒加载版本）"""

    def __init__(self, ecgqa_dataset, target_seq_len=1000, scaler=None, scaler_sample_size=1000):
        """
        Args:
            ecgqa_dataset: ECGQAClassificationDataset实例
            target_seq_len: 目标序列长度（采样点数）
            scaler: StandardScaler对象（如果为None，将从数据中拟合）
            scaler_sample_size: 用于拟合scaler的样本数量（默认1000）
        """
        self.ecgqa_dataset = ecgqa_dataset
        self.target_seq_len = target_seq_len
        self.template_id = ecgqa_dataset.template_id
        self.classification_info = ecgqa_dataset.classification_info
        self.input_ecg_num = self.classification_info['input_ecg_num']
        self.output_dim = self.classification_info['output_dim']
        self.output_classes = sorted(
            self.classification_info['output_classes'])

        print(f"准备template_id={self.template_id}的数据集（懒加载模式）...")

        # 第一遍遍历：收集样本元数据（不加载ECG信号）并计算scaler
        print("收集样本元数据...")
        self.sample_metadata = []  # 存储样本的元数据（ecg_path, answer等）
        actual_input_ecg_num = None
        sample_count = 0

        for sample in tqdm(ecgqa_dataset, desc=f"Collecting metadata for template {self.template_id}"):
            # 从第一个样本获取actual_input_ecg_num
            if actual_input_ecg_num is None:
                actual_input_ecg_num = len(sample['ecg_path'])
                print(
                    f"实际输入ECG数量: {actual_input_ecg_num} (classification_info: {self.input_ecg_num})")

            # 只存储元数据，不存储ECG信号
            metadata = {
                'ecg_path': sample['ecg_path'],
                'answer': sample['answer'],
                'template_answer': sorted(sample['template_answer'])
            }
            self.sample_metadata.append(metadata)
            sample_count += 1

        if actual_input_ecg_num is None:
            raise ValueError(f"Template {self.template_id} 没有找到任何样本")

        self.actual_input_ecg_num = actual_input_ecg_num
        print(f"找到 {len(self.sample_metadata)} 个样本")

        # 计算scaler（采样计算以节省内存）
        if scaler is None:
            print("计算StandardScaler（采样计算）...")
            self.scaler = StandardScaler()
            scaler_samples = []
            sample_indices = np.linspace(0, len(self.sample_metadata) - 1,
                                         min(scaler_sample_size, len(
                                             self.sample_metadata)),
                                         dtype=int)

            for idx in tqdm(sample_indices, desc="Fitting scaler"):
                sample = self._load_sample(idx, apply_scaling=False)  # 不应用标准化
                scaler_samples.append(sample)

            scaler_data = np.vstack(scaler_samples).reshape(-1, 12)
            self.scaler.fit(scaler_data)
            print("StandardScaler计算完成")
        else:
            self.scaler = scaler

    def _load_sample(self, idx, apply_scaling=True):
        """加载单个样本的ECG信号并处理（懒加载）

        Args:
            idx: 样本索引
            apply_scaling: 是否应用标准化（在拟合scaler时应为False）
        """
        metadata = self.sample_metadata[idx]
        ecg_paths = metadata['ecg_path']

        # 加载ECG信号
        ecg_signals = []
        for ecg_path in ecg_paths:
            signals, fields = wfdb.rdsamp(ecg_path)
            ecg_data = EcgSignals(signals=signals, fields=fields)
            ecg_signals.append(ecg_data)

        # 根据actual_input_ecg_num处理输入
        if self.actual_input_ecg_num is not None:
            if len(ecg_signals) != self.actual_input_ecg_num:
                if len(ecg_signals) < self.actual_input_ecg_num:
                    while len(ecg_signals) < self.actual_input_ecg_num:
                        ecg_signals.append(ecg_signals[-1])
                else:
                    ecg_signals = ecg_signals[:self.actual_input_ecg_num]

        # 处理每个ECG信号：调整长度
        processed_signals = []
        for ecg_signal in ecg_signals:
            signal = ecg_signal.signals  # (n_samples, n_channels)

            # 确保是12导联
            if signal.shape[1] != 12:
                raise ValueError(f"ECG信号应该有12个导联，但得到 {signal.shape[1]} 个")

            # 调整长度到target_seq_len
            current_length = signal.shape[0]
            if current_length != self.target_seq_len:
                if current_length > self.target_seq_len:
                    # 裁剪到目标长度（取中间部分）
                    start_idx = (current_length - self.target_seq_len) // 2
                    signal = signal[start_idx:start_idx + self.target_seq_len]
                else:
                    # 填充到目标长度（在两端填充零）
                    pad_size = self.target_seq_len - current_length
                    pad_before = pad_size // 2
                    pad_after = pad_size - pad_before
                    signal = np.pad(
                        signal, ((pad_before, pad_after), (0, 0)),
                        mode='constant', constant_values=0
                    )

            processed_signals.append(signal)

        # 如果有多个ECG信号，拼接后下采样
        if self.actual_input_ecg_num is not None and self.actual_input_ecg_num > 1:
            combined_signal = np.concatenate(processed_signals, axis=0)
            if combined_signal.shape[0] > self.target_seq_len:
                step = combined_signal.shape[0] // self.target_seq_len
                combined_signal = combined_signal[::step][:self.target_seq_len]
            elif combined_signal.shape[0] < self.target_seq_len:
                pad_size = self.target_seq_len - combined_signal.shape[0]
                pad_before = pad_size // 2
                pad_after = pad_size - pad_before
                combined_signal = np.pad(
                    combined_signal, ((pad_before, pad_after), (0, 0)),
                    mode='constant', constant_values=0
                )
            final_signal = combined_signal
        else:
            final_signal = processed_signals[0]

        # 标准化（仅在apply_scaling=True且scaler已fit时应用）
        if apply_scaling and hasattr(self.scaler, 'mean_'):
            signal_shape = final_signal.shape
            final_signal = self.scaler.transform(
                final_signal.flatten()[:, np.newaxis]).reshape(signal_shape)

        return final_signal

    def _get_label(self, idx):
        """获取样本的onehot标签"""
        metadata = self.sample_metadata[idx]
        answer = metadata['answer']
        template_answer = metadata['template_answer']

        # 创建onehot编码
        onehot = np.zeros(self.output_dim, dtype=np.float32)
        for ans in answer:
            if ans in template_answer:
                idx_label = template_answer.index(ans)
                onehot[idx_label] = 1.0

        return onehot

    def __len__(self):
        return len(self.sample_metadata)

    def __getitem__(self, idx):
        """按需加载样本（懒加载）"""
        signal = self._load_sample(idx)
        label = self._get_label(idx)
        return torch.FloatTensor(signal), torch.FloatTensor(label)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for X, y in tqdm(dataloader, desc="Training"):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Validating"):
            X, y = X.to(device), y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    if len(all_preds) == 0:
        return 0.0, 0.0, 0.0, 0.0

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # 计算指标
    avg_loss = total_loss / len(dataloader)

    # 对于多标签分类，使用阈值0.5进行二值化
    preds_binary = (all_preds > 0.5).astype(int)

    # 计算macro AUC
    try:
        macro_auc = roc_auc_score(all_labels, all_preds, average='macro')
    except:
        macro_auc = 0.0

    # 计算准确率（完全匹配）
    accuracy = accuracy_score(all_labels, preds_binary)

    # 计算F1分数
    try:
        f1 = f1_score(all_labels, preds_binary, average='macro')
    except:
        f1 = 0.0

    return avg_loss, macro_auc, accuracy, f1


def save_checkpoint(model, optimizer, epoch, loss, auc, checkpoint_dir, template_id, is_best=False):
    """保存checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'auc': auc,
        'template_id': template_id
    }

    # 保存最新checkpoint
    checkpoint_path = os.path.join(
        checkpoint_dir, f'template_{template_id}_latest_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)

    # 保存最佳checkpoint
    if is_best:
        best_path = os.path.join(
            checkpoint_dir, f'template_{template_id}_best_checkpoint.pth')
        torch.save(checkpoint, best_path)
        print(f"保存最佳模型: {best_path} (AUC: {auc:.4f})")

    # 每5个epoch保存一次
    if (epoch + 1) % 5 == 0:
        epoch_path = os.path.join(
            checkpoint_dir, f'template_{template_id}_checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, epoch_path)
        print(f"保存epoch {epoch+1} checkpoint: {epoch_path}")


def train_template_model(template_id, dataset, args, output_base_dir):
    """为单个template_id训练模型"""
    print("\n" + "=" * 80)
    print(f"训练 Template ID: {template_id}")
    print("=" * 80)

    # 创建输出目录
    template_output_dir = os.path.join(
        output_base_dir, f'template_{template_id}')
    checkpoint_dir = os.path.join(template_output_dir, 'checkpoints')
    data_dir = os.path.join(template_output_dir, 'data')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # 获取分类信息
    classification_info = dataset.classification_info
    input_ecg_num = classification_info['input_ecg_num']
    output_dim = classification_info['output_dim']
    output_classes = sorted(classification_info['output_classes'])

    print(f"输入ECG数量: {input_ecg_num}")
    print(f"输出维度: {output_dim}")
    print(f"输出类别: {output_classes}")

    # 创建数据集包装器
    print("\n准备数据集...")
    full_dataset = ECGQAClassificationDatasetWrapper(
        dataset,
        target_seq_len=args.seq_len,
        scaler=None
    )

    # 保存scaler
    scaler_path = os.path.join(data_dir, 'standard_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(full_dataset.scaler, f)

    # 保存类别信息
    class_info = {
        'template_id': template_id,
        'input_ecg_num': input_ecg_num,
        'output_dim': output_dim,
        'output_classes': output_classes,
        'classification_info': classification_info
    }
    class_info_path = os.path.join(data_dir, 'class_info.json')
    with open(class_info_path, 'w') as f:
        json.dump(class_info, f, indent=2)

    # 分割数据集（80%训练，20%验证）
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    if dataset_size < 10:
        print(f"警告: 数据集太小 ({dataset_size} 个样本)，跳过训练")
        return

    # 使用随机种子确保可重复性
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"训练集: {train_size} 个样本")
    print(f"验证集: {val_size} 个样本")

    # 创建模型
    input_shape = (args.seq_len, 12)  # (seq_len, n_channels)
    model = create_inception_time_model(
        input_shape=input_shape,
        n_classes=output_dim,
        depth=args.depth,
        kernel_size=args.kernel_size,
        bottleneck_size=args.bottleneck_size,
        nb_filters=args.nb_filters,
        use_residual=True
    )

    device = torch.device(args.device)
    model = model.to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数数量: {total_params:,} (可训练: {trainable_params:,})")

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练循环
    best_val_auc = 0.0
    best_val_loss = float('inf')

    print("\n开始训练...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)

        # 训练
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, val_auc, val_acc, val_f1 = validate(
            model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(
            f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # 保存checkpoint
        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            best_val_loss = val_loss

        save_checkpoint(model, optimizer, epoch, val_loss,
                        val_auc, checkpoint_dir, template_id, is_best)

    # 保存最终模型
    final_model_path = os.path.join(
        checkpoint_dir, f'template_{template_id}_final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_shape': input_shape,
        'n_classes': output_dim,
        'depth': args.depth,
        'kernel_size': args.kernel_size,
        'bottleneck_size': args.bottleneck_size,
        'nb_filters': args.nb_filters,
        'template_id': template_id,
        'classification_info': classification_info
    }, final_model_path)
    print(f"\n最终模型已保存: {final_model_path}")

    print(f"\nTemplate {template_id} 训练完成！")


def main():
    parser = argparse.ArgumentParser(description='训练Inception Time模型用于ECGQA分类')
    parser.add_argument('--dataset_dir', type=str,
                        default='dataset/ecgqa_ptbxl/template/train',
                        help='训练数据集目录')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--seq_len', type=int, default=1000, help='序列长度（采样点数）')
    parser.add_argument('--depth', type=int, default=9, help='模型深度')
    parser.add_argument('--kernel_size', type=int, default=60, help='卷积核大小')
    parser.add_argument('--bottleneck_size', type=int,
                        default=32, help='Bottleneck大小')
    parser.add_argument('--nb_filters', type=int, default=32, help='滤波器数量')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--num_workers', type=int,
                        default=4, help='数据加载器工作进程数')
    parser.add_argument('--template_ids', type=str, default=None,
                        help='要训练的template_id列表（逗号分隔），如果为None则训练所有')

    args = parser.parse_args()

    print("=" * 80)
    print("训练Inception Time模型用于ECGQA分类")
    print("=" * 80)
    print(f"数据集目录: {args.dataset_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"序列长度: {args.seq_len}")
    print(f"设备: {args.device}")
    print("=" * 80)

    # 创建ECGQAClassificationManager
    print("\n初始化ECGQAClassificationManager...")
    eqcm = ECGQAClassificationManager(dataset_dir=args.dataset_dir)

    print(f"找到 {len(eqcm.dataset_dict)} 个template")

    # 确定要训练的template_id列表
    if args.template_ids:
        template_ids = [int(tid) for tid in args.template_ids.split(',')]
        # 过滤出存在的template_id
        template_ids = [
            tid for tid in template_ids if tid in eqcm.dataset_dict]
        print(f"指定训练 {len(template_ids)} 个template")
    else:
        template_ids = list(eqcm.dataset_dict.keys())
        print(f"将训练所有 {len(template_ids)} 个template")

    # 创建输出目录
    output_base_dir = os.path.join(args.output_dir, 'templates')
    os.makedirs(output_base_dir, exist_ok=True)

    # 为每个template_id训练模型
    for template_id in template_ids:
        try:
            dataset = eqcm.dataset_dict[template_id]
            train_template_model(template_id, dataset, args, output_base_dir)
        except Exception as e:
            print(f"\n训练template {template_id}时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("所有模板训练完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
