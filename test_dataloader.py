#!/usr/bin/env python3
"""
测试程序：读取ECG-QA数据集的前20个样本
"""

import os
import sys
import numpy as np
from data.EcgQaData import EcgQaDataset


def test_ecg_qa_dataset():
    """测试EcgQaDataset，读取前20个数据样本"""
    
    # 数据集路径
    dataset_dir = "/home/xl/agentECG/dataset/ecgqa_ptbxl/paraphrased/train"
    
    print(f"正在测试数据集: {dataset_dir}")
    print("="*60)
    
    # 创建数据集
    try:
        dataset = EcgQaDataset(dataset_dir)
        print("✓ 数据集创建成功")
    except Exception as e:
        print(f"✗ 数据集创建失败: {e}")
        return
    
    # 读取前20个样本
    sample_count = 0
    max_samples = 20
    
    print(f"\n开始读取前 {max_samples} 个样本...")
    print("-"*60)
    
    try:
        for i, sample in enumerate(dataset):
            sample_count += 1
            
            print(f"\n样本 {sample_count}:")
            print(f"  Template ID: {sample.get('template_id', 'N/A')}")
            print(f"  Question ID: {sample.get('question_id', 'N/A')}")
            print(f"  Sample ID: {sample.get('sample_id', 'N/A')}")
            print(f"  ECG ID: {sample.get('ecg_id', 'N/A')}")
            print(f"  Question Type: {sample.get('question_type', 'N/A')}")
            print(f"  Attribute Type: {sample.get('attribute_type', 'N/A')}")
            print(f"  Question: {sample.get('question', 'N/A')[:100]}...")  # 只显示前100个字符
            print(f"  Answer: {sample.get('answer', 'N/A')}")
            print(f"  Attribute: {sample.get('attribute', 'N/A')}")
            print(f"  ECG Path: {sample.get('ecg_path', 'N/A')}")
            
            # 检查ECG数据
            if 'ecg_data' in sample and sample['ecg_data'] is not None:
                ecg_data = sample['ecg_data']
                signals, fields = ecg_data.get_data()
                
                print(f"  ECG信号形状: {signals.shape}")
                print(f"  采样频率: {fields.get('fs', 'N/A')} Hz")
                print(f"  导联数量: {fields.get('n_sig', 'N/A')}")
                print(f"  信号长度: {fields.get('sig_len', 'N/A')}")
                print(f"  导联名称: {fields.get('sig_name', 'N/A')}")
                print(f"  信号范围: [{np.min(signals):.3f}, {np.max(signals):.3f}]")
                
                # 检查是否有异常值
                if np.any(np.isnan(signals)):
                    print(f"  ⚠️  警告: 信号中包含NaN值")
                if np.any(np.isinf(signals)):
                    print(f"  ⚠️  警告: 信号中包含无穷大值")
                    
            else:
                print(f"  ✗ ECG数据加载失败")
            
            # 达到最大样本数就停止
            if sample_count >= max_samples:
                break
                
    except Exception as e:
        print(f"✗ 读取样本时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print(f"测试完成！成功读取了 {sample_count} 个样本")
    
    # 统计信息
    if sample_count > 0:
        print(f"\n数据集统计:")
        print(f"  总样本数 (前{max_samples}个): {sample_count}")
        print(f"  数据加载: {'成功' if sample_count == max_samples else '部分成功'}")


def test_single_sample():
    """测试单个样本的详细信息"""
    
    dataset_dir = "/home/xl/agentECG/dataset/ecgqa_ptbxl/paraphrased/train"
    
    print("\n" + "="*60)
    print("详细测试单个样本")
    print("="*60)
    
    try:
        dataset = EcgQaDataset(dataset_dir)
        
        # 获取第一个样本
        sample_iter = iter(dataset)
        sample = next(sample_iter)
        
        print("样本详细信息:")
        for key, value in sample.items():
            if key == 'ecg_data':
                if value is not None:
                    signals, fields = value.get_data()
                    print(f"  {key}:")
                    print(f"    - signals shape: {signals.shape}")
                    print(f"    - signals dtype: {signals.dtype}")
                    print(f"    - fields keys: {list(fields.keys())}")
                else:
                    print(f"  {key}: None")
            elif isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")
        
        # ECG信号的详细分析
        if 'ecg_data' in sample and sample['ecg_data'] is not None:
            ecg_data = sample['ecg_data']
            signals, fields = ecg_data.get_data()
            
            print(f"\nECG信号详细分析:")
            print(f"  信号形状: {signals.shape}")
            print(f"  信号类型: {type(signals)}")
            print(f"  数据类型: {signals.dtype}")
            
            # 每个导联的统计信息
            for i, lead_name in enumerate(fields.get('sig_name', [])):
                lead_signal = signals[:, i]
                print(f"  导联 {lead_name}:")
                print(f"    - 均值: {np.mean(lead_signal):.3f}")
                print(f"    - 标准差: {np.std(lead_signal):.3f}")
                print(f"    - 最小值: {np.min(lead_signal):.3f}")
                print(f"    - 最大值: {np.max(lead_signal):.3f}")
                
                if i >= 2:  # 只显示前3个导联
                    print(f"  ... (还有{len(fields.get('sig_name', []))-3}个导联)")
                    break
        
    except Exception as e:
        print(f"✗ 单样本测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("ECG-QA 数据集测试程序")
    print("=" * 60)
    
    # 检查数据集目录是否存在
    dataset_dir = "/home/xl/agentECG/dataset/ecgqa_ptbxl/paraphrased/train"
    if not os.path.exists(dataset_dir):
        print(f"✗ 数据集目录不存在: {dataset_dir}")
        sys.exit(1)
    
    # 运行测试
    test_ecg_qa_dataset()
    test_single_sample()