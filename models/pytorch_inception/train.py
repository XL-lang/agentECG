"""
训练Inception Time模型
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

from data_loader import prepare_data, PTBXLDataset
from inception_time import create_inception_time_model


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
    
    return total_loss / num_batches


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
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # 计算macro AUC
    try:
        macro_auc = roc_auc_score(all_labels, all_preds, average='macro')
    except:
        macro_auc = 0.0
    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, macro_auc


def save_checkpoint(model, optimizer, epoch, loss, auc, checkpoint_dir, is_best=False):
    """保存checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'auc': auc
    }
    
    # 保存最新checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最佳checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
        torch.save(checkpoint, best_path)
        print(f"保存最佳模型: {best_path} (AUC: {auc:.4f})")
    
    # 每5个epoch保存一次
    if (epoch + 1) % 5 == 0:
        epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, epoch_path)
        print(f"保存epoch {epoch+1} checkpoint: {epoch_path}")


def main():
    parser = argparse.ArgumentParser(description='训练Inception Time模型')
    parser.add_argument('--datafolder', type=str, 
                       default='/home/xl/dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/',
                       help='数据集文件夹路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='输出目录')
    parser.add_argument('--task', type=str, default='all', 
                       choices=['all', 'diagnostic', 'subdiagnostic', 'superdiagnostic', 'form', 'rhythm'],
                       help='任务类型')
    parser.add_argument('--sampling_rate', type=int, default=100, choices=[100, 500],
                       help='采样率')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--depth', type=int, default=9, help='模型深度')
    parser.add_argument('--kernel_size', type=int, default=60, help='卷积核大小')
    parser.add_argument('--bottleneck_size', type=int, default=32, help='Bottleneck大小')
    parser.add_argument('--nb_filters', type=int, default=32, help='滤波器数量')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("训练Inception Time模型")
    print("=" * 60)
    print(f"数据集: {args.datafolder}")
    print(f"任务: {args.task}")
    print(f"采样率: {args.sampling_rate} Hz")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"模型深度: {args.depth}")
    print(f"卷积核大小: {args.kernel_size}")
    print(f"设备: {args.device}")
    print("=" * 60)
    
    # 准备数据
    output_data_dir = os.path.join(args.output_dir, 'data')
    os.makedirs(output_data_dir, exist_ok=True)
    
    X_train, X_val, X_test, y_train, y_val, y_test, n_classes, mlb = prepare_data(
        datafolder=args.datafolder,
        task=args.task,
        sampling_rate=args.sampling_rate,
        output_dir=output_data_dir
    )
    
    # 创建数据集和数据加载器
    train_dataset = PTBXLDataset(X_train, y_train)
    val_dataset = PTBXLDataset(X_val, y_val)
    test_dataset = PTBXLDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    
    # 创建模型
    input_shape = (X_train.shape[1], X_train.shape[2])  # (seq_len, n_channels)
    model = create_inception_time_model(
        input_shape=input_shape,
        n_classes=n_classes,
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练循环
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_auc = 0.0
    best_val_loss = float('inf')
    
    print("\n开始训练...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        # 保存checkpoint
        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            best_val_loss = val_loss
        
        save_checkpoint(model, optimizer, epoch, val_loss, val_auc, checkpoint_dir, is_best)
    
    # 训练完成，在测试集上评估
    print("\n" + "=" * 60)
    print("训练完成！在测试集上评估...")
    print("=" * 60)
    
    # 加载最佳模型
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载最佳模型 (epoch {checkpoint['epoch']+1}, AUC: {checkpoint['auc']:.4f})")
    
    test_loss, test_auc = validate(model, test_loader, criterion, device)
    print(f"\n测试集结果:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test AUC: {test_auc:.4f}")
    
    # 保存最终模型
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_shape': input_shape,
        'n_classes': n_classes,
        'depth': args.depth,
        'kernel_size': args.kernel_size,
        'bottleneck_size': args.bottleneck_size,
        'nb_filters': args.nb_filters,
        'test_auc': test_auc,
        'test_loss': test_loss
    }, final_model_path)
    print(f"\n最终模型已保存: {final_model_path}")
    
    print("\n训练完成！")


if __name__ == '__main__':
    main()

