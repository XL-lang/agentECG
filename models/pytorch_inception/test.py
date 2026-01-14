"""
测试Inception Time模型
"""
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from data_loader import prepare_data, PTBXLDataset
from inception_time import create_inception_time_model


def evaluate_model(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # 计算macro AUC
    try:
        macro_auc = roc_auc_score(all_labels, all_preds, average='macro')
    except Exception as e:
        print(f"计算macro AUC时出错: {e}")
        macro_auc = 0.0
    
    # 计算micro AUC
    try:
        micro_auc = roc_auc_score(all_labels, all_preds, average='micro')
    except:
        micro_auc = 0.0
    
    # 计算每个类别的AUC
    class_aucs = []
    for i in range(all_labels.shape[1]):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            class_aucs.append(auc)
        except:
            class_aucs.append(0.0)
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'macro_auc': macro_auc,
        'micro_auc': micro_auc,
        'class_aucs': class_aucs
    }


def main():
    parser = argparse.ArgumentParser(description='测试Inception Time模型')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--datafolder', type=str,
                       default='/home/xl/dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/',
                       help='数据集文件夹路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='输出目录（用于加载scaler和mlb）')
    parser.add_argument('--task', type=str, default='all',
                       choices=['all', 'diagnostic', 'subdiagnostic', 'superdiagnostic', 'form', 'rhythm'],
                       help='任务类型')
    parser.add_argument('--sampling_rate', type=int, default=100, choices=[100, 500],
                       help='采样率')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--save_predictions', action='store_true',
                       help='是否保存预测结果')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("测试Inception Time模型")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"数据集: {args.datafolder}")
    print(f"任务: {args.task}")
    print(f"设备: {args.device}")
    print("=" * 60)
    
    # 加载checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    device = torch.device(args.device)
    
    # 准备数据
    output_data_dir = os.path.join(args.output_dir, 'data')
    X_train, X_val, X_test, y_train, y_val, y_test, n_classes, mlb = prepare_data(
        datafolder=args.datafolder,
        task=args.task,
        sampling_rate=args.sampling_rate,
        output_dir=output_data_dir
    )
    
    # 创建数据集和数据加载器
    test_dataset = PTBXLDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    if 'input_shape' in checkpoint:
        input_shape = checkpoint['input_shape']
        depth = checkpoint.get('depth', 9)
        kernel_size = checkpoint.get('kernel_size', 60)
        bottleneck_size = checkpoint.get('bottleneck_size', 32)
        nb_filters = checkpoint.get('nb_filters', 32)
    else:
        # 从数据推断
        input_shape = (X_test.shape[1], X_test.shape[2])
        depth = 9
        kernel_size = 60
        bottleneck_size = 32
        nb_filters = 32
    
    model = create_inception_time_model(
        input_shape=input_shape,
        n_classes=n_classes,
        depth=depth,
        kernel_size=kernel_size,
        bottleneck_size=bottleneck_size,
        nb_filters=nb_filters,
        use_residual=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"\n模型参数:")
    print(f"  输入形状: {input_shape}")
    print(f"  类别数: {n_classes}")
    print(f"  深度: {depth}")
    print(f"  卷积核大小: {kernel_size}")
    
    # 评估模型
    print("\n在测试集上评估...")
    results = evaluate_model(model, test_loader, device)
    
    print(f"\n测试集结果:")
    print(f"  Macro AUC: {results['macro_auc']:.4f}")
    print(f"  Micro AUC: {results['micro_auc']:.4f}")
    print(f"  各类别AUC: {np.mean(results['class_aucs']):.4f} (平均)")
    
    # 保存预测结果
    if args.save_predictions:
        pred_dir = os.path.join(args.output_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        
        np.save(os.path.join(pred_dir, 'y_test_pred.npy'), results['predictions'])
        np.save(os.path.join(pred_dir, 'y_test_true.npy'), results['labels'])
        print(f"\n预测结果已保存到: {pred_dir}")
    
    print("\n测试完成！")


if __name__ == '__main__':
    main()

