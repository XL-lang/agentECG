"""
验证模型是否可以正常创建和运行
"""
import torch
from inception_time import create_inception_time_model

def test_model():
    """测试模型创建和前向传播"""
    print("测试Inception Time模型...")
    
    # 模型参数（对应all任务的配置）
    input_shape = (1000, 12)  # (seq_len, n_channels)
    n_classes = 71
    depth = 9
    kernel_size = 60
    bottleneck_size = 32
    nb_filters = 32
    
    # 创建模型
    print(f"创建模型: input_shape={input_shape}, n_classes={n_classes}, depth={depth}")
    model = create_inception_time_model(
        input_shape=input_shape,
        n_classes=n_classes,
        depth=depth,
        kernel_size=kernel_size,
        bottleneck_size=bottleneck_size,
        nb_filters=nb_filters,
        use_residual=True
    )
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, *input_shape)
    print(f"输入形状: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 验证输出形状
    assert output.shape == (batch_size, n_classes), f"输出形状错误: {output.shape} != ({batch_size}, {n_classes})"
    assert (output >= 0).all() and (output <= 1).all(), "输出应该在[0, 1]范围内（sigmoid）"
    
    print("\n✅ 模型测试通过！")
    return model

if __name__ == '__main__':
    test_model()

