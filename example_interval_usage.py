"""
使用示例：如何使用 EcgSignals.interpolate_intervals() 方法

该方法的功能：
1. 根据时间距离匹配 P wave、QRS complex 和 T wave（允许一个波形被多次匹配）
2. 计算三种间期/段：
   - PR interval: 从 P 波开始到 QRS 波开始
   - QT interval: 从 QRS 波开始到 T 波结束
   - ST segment: 从 QRS 波结束到 T 波开始
3. 使用 40% 阈值去除离群值（基于间期长度）
"""

import numpy as np
import sys
sys.path.insert(0, '/home/xl/agentECG')

# 示例 1: 基本使用
print("="*70)
print("示例 1: 基本使用")
print("="*70)

# 假设你已经从 extract_ecg_wave_segments 获得了分段结果
segments = {
    'P wave': [(37, 80), (1044, 1146), (2328, 2431), (3643, 3747), (4969, 5000)],
    'QRS complex': [(402, 508), (1665, 1769), (3000, 3102), (4324, 4429)],
    'T wave': [(402, 508), (1665, 1769), (3000, 3102), (4324, 4429), (4969, 5000)]
}

# 创建 EcgSignals 对象（这里使用简化版本进行演示）
from test_intervals import EcgSignalsTest
ecg = EcgSignalsTest()

# 调用 interpolate_intervals 方法
result = ecg.interpolate_intervals(segments)

# 查看结果
print("\n结果:")
for interval_type, intervals in result.items():
    if interval_type == 'matched_beats':
        continue
    print(f"\n{interval_type}:")
    for i, (start, end) in enumerate(intervals, 1):
        length = end - start
        print(f"  第 {i} 个: 索引 [{start}, {end}], 长度 = {length}")

print("\n\n" + "="*70)
print("示例 2: 包含离群值的数据")
print("="*70)

# 添加一个异常的 QRS complex（长度远大于其他）
segments_with_outliers = {
    'P wave': [(37, 80), (1044, 1146), (2328, 2431), (3643, 3747), (4000, 4100), (5000, 5100)],
    'QRS complex': [(402, 508), (1665, 1769), (3000, 3102), (4324, 5500), (5200, 5300)],  # (4324, 5500) 是离群值
    'T wave': [(600, 700), (1900, 2000), (3200, 3300), (5600, 5700), (5400, 5500)]
}

result_with_outliers = ecg.interpolate_intervals(segments_with_outliers)

print("\n原始 QRS complexes:", segments_with_outliers['QRS complex'])
print("\n过滤后的 PR intervals 数量:", len(result_with_outliers['PR interval']))
print("过滤后的 QT intervals 数量:", len(result_with_outliers['QT interval']))
print("过滤后的 ST segments 数量:", len(result_with_outliers['ST segment']))

# 显示详细信息
print("\nPR intervals (离群值已被移除):")
for i, (start, end) in enumerate(result_with_outliers['PR interval'], 1):
    length = end - start
    print(f"  第 {i} 个: 索引 [{start}, {end}], 长度 = {length}")

print("\n\n" + "="*70)
print("示例 3: 计算间期的统计信息")
print("="*70)

def calculate_interval_stats(intervals, fs=500):
    """
    计算间期的统计信息
    
    Args:
        intervals: list of (start, end) tuples
        fs: 采样频率 (Hz)
    
    Returns:
        dict: 包含统计信息的字典
    """
    if not intervals:
        return None
    
    lengths = [end - start for start, end in intervals]
    lengths_ms = [length / fs * 1000 for length in lengths]  # 转换为毫秒
    
    return {
        'count': len(intervals),
        'mean_samples': np.mean(lengths),
        'std_samples': np.std(lengths),
        'median_samples': np.median(lengths),
        'mean_ms': np.mean(lengths_ms),
        'std_ms': np.std(lengths_ms),
        'median_ms': np.median(lengths_ms),
        'min_ms': np.min(lengths_ms),
        'max_ms': np.max(lengths_ms)
    }

result = ecg.interpolate_intervals(segments)

print("\n间期统计信息 (采样频率 = 500 Hz):\n")
for interval_type in ['PR interval', 'QT interval', 'ST segment']:
    stats = calculate_interval_stats(result[interval_type], fs=500)
    if stats:
        print(f"{interval_type}:")
        print(f"  数量: {stats['count']}")
        print(f"  平均值: {stats['mean_ms']:.2f} ms ({stats['mean_samples']:.1f} 样本)")
        print(f"  标准差: {stats['std_ms']:.2f} ms")
        print(f"  中位数: {stats['median_ms']:.2f} ms")
        print(f"  范围: [{stats['min_ms']:.2f}, {stats['max_ms']:.2f}] ms")
        print()

print("\n" + "="*70)
print("所有示例完成！")
print("="*70)

# 临床参考值
print("\n临床参考值 (正常成人):")
print("  PR interval: 120-200 ms")
print("  QT interval: 350-450 ms (取决于心率)")
print("  ST segment: 通常 80-120 ms")
