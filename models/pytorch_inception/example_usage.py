"""
InceptionClassifier使用示例
"""
import numpy as np
from classifier import InceptionClassifier

# 示例1: 基本使用
def example_basic():
    print("=" * 60)
    print("示例1: 基本使用")
    print("=" * 60)
    
    # 加载分类器
    classifier = InceptionClassifier(
        checkpoint_path='./output/checkpoints/best_checkpoint.pth',
        data_dir='./output/data'
    )
    
    # 创建示例ECG信号（1000个时间点，12个导联）
    ecg_signal = np.random.randn(1000, 12).astype(np.float32)
    
    # 预测（只返回置信度>0.5的类别）
    result = classifier.predict(ecg_signal)
    
    print(f"\n预测结果（置信度>0.5的类别，共{len(result)}个）:")
    for i, (class_name, confidence) in enumerate(list(result.items())[:10]):
        print(f"  {i+1}. {class_name}: {confidence:.4f}")
    if len(result) > 10:
        print(f"  ... 还有 {len(result) - 10} 个类别")


# 示例2: 获取所有类别的概率
def example_all_probs():
    print("\n" + "=" * 60)
    print("示例2: 获取所有类别的概率")
    print("=" * 60)
    
    classifier = InceptionClassifier(
        checkpoint_path='./output/checkpoints/best_checkpoint.pth',
        data_dir='./output/data'
    )
    
    ecg_signal = np.random.randn(1000, 12).astype(np.float32)
    
    # 获取所有类别的概率
    all_probs = classifier.predict_proba(ecg_signal)
    
    # 按置信度排序
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n所有类别概率（前10个）:")
    for i, (class_name, confidence) in enumerate(sorted_probs[:10]):
        print(f"  {i+1}. {class_name}: {confidence:.4f}")
    
    print(f"\n总类别数: {len(all_probs)}")
    print(f"置信度>0.5的类别数: {len([v for v in all_probs.values() if v > 0.5])}")
    print(f"置信度>0.3的类别数: {len([v for v in all_probs.values() if v > 0.3])}")


# 示例3: 批量预测
def example_batch():
    print("\n" + "=" * 60)
    print("示例3: 批量预测")
    print("=" * 60)
    
    classifier = InceptionClassifier(
        checkpoint_path='./output/checkpoints/best_checkpoint.pth',
        data_dir='./output/data'
    )
    
    # 创建多个ECG信号（5个样本）
    ecg_signals = np.random.randn(5, 1000, 12).astype(np.float32)
    
    # 批量预测
    results = classifier.predict(ecg_signals)
    
    print(f"\n批量预测结果（{len(results)}个样本）:")
    for i, result in enumerate(results):
        print(f"\n样本 {i+1}:")
        print(f"  检测到 {len(result)} 个类别（置信度>0.5）")
        if len(result) > 0:
            top_class = max(result.items(), key=lambda x: x[1])
            print(f"  最高置信度: {top_class[0]} ({top_class[1]:.4f})")


# 示例4: 处理不同长度的ECG信号
def example_different_lengths():
    print("\n" + "=" * 60)
    print("示例4: 处理不同长度的ECG信号")
    print("=" * 60)
    
    classifier = InceptionClassifier(
        checkpoint_path='./output/checkpoints/best_checkpoint.pth',
        data_dir='./output/data'
    )
    
    # 不同长度的ECG信号
    lengths = [500, 1000, 2000, 3000]
    
    for length in lengths:
        ecg_signal = np.random.randn(length, 12).astype(np.float32)
        result = classifier.predict(ecg_signal)
        
        print(f"\n长度 {length}:")
        print(f"  检测到 {len(result)} 个类别（置信度>0.5）")
        if len(result) > 0:
            top_class = max(result.items(), key=lambda x: x[1])
            print(f"  最高置信度: {top_class[0]} ({top_class[1]:.4f})")


if __name__ == '__main__':
    import sys
    
    # 检查checkpoint是否存在
    checkpoint_path = './output/checkpoints/best_checkpoint.pth'
    data_dir = './output/data'
    
    import os
    if not os.path.exists(checkpoint_path):
        print(f"错误: 未找到checkpoint文件: {checkpoint_path}")
        print("请先训练模型或指定正确的checkpoint路径")
        sys.exit(1)
    
    if not os.path.exists(data_dir):
        print(f"错误: 未找到数据目录: {data_dir}")
        print("请确保数据目录包含mlb.pkl和standard_scaler.pkl文件")
        sys.exit(1)
    
    # 运行示例
    try:
        example_basic()
        example_all_probs()
        example_batch()
        example_different_lengths()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

