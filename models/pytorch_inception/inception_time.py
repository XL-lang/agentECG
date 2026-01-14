"""
Inception Time模型实现（PyTorch版本）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """Inception模块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=40, bottleneck_size=32, stride=1, activation='linear'):
        super(InceptionModule, self).__init__()
        
        self.use_bottleneck = bottleneck_size > 0 and in_channels > 1
        
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, padding=0, bias=False)
            in_channels = bottleneck_size
        
        # 计算不同大小的kernel
        kernel_sizes = [kernel_size // (2 ** i) for i in range(3)]
        
        # 多个不同kernel size的卷积
        # 对于stride=1的Conv1d，使用padding来保证输出长度尽可能接近输入长度
        # padding = (kernel_size - 1) // 2 可以保证奇数kernel_size时输出长度=输入长度
        # 对于偶数kernel_size，输出长度会少1，我们稍后在forward中处理
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=ks, 
                     stride=stride, padding=(ks-1)//2, bias=False)
            for ks in kernel_sizes
        ])
        
        # MaxPooling + Conv1x1
        # 对于stride=1，padding=1可以保持长度不变（kernel_size=3时）
        pool_padding = 1 if stride == 1 else 0
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=stride, padding=pool_padding)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        
        self.bn = nn.BatchNorm1d(out_channels * 4)  # 3个conv + 1个pool conv
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x):
        if self.use_bottleneck:
            x = self.bottleneck(x)
        
        # 获取目标长度（使用pool_out的长度作为参考，因为它应该和输入长度一致）
        pool_out = self.maxpool(x)
        target_length = pool_out.shape[2]
        
        # 多个不同kernel size的卷积
        conv_outputs = []
        for conv in self.convs:
            out = conv(x)
            # 如果输出长度不一致，进行裁剪或填充以对齐
            current_length = out.shape[2]
            if current_length != target_length:
                if current_length > target_length:
                    # 裁剪到目标长度
                    out = out[:, :, :target_length]
                else:
                    # 填充到目标长度（在末尾填充零）
                    pad_size = target_length - current_length
                    out = torch.nn.functional.pad(out, (0, pad_size), mode='constant', value=0)
            conv_outputs.append(out)
        
        # MaxPooling + Conv1x1
        pool_out = self.conv_pool(pool_out)
        
        # 确保pool_out的长度也一致（虽然理论上应该已经一致了）
        if pool_out.shape[2] != target_length:
            if pool_out.shape[2] > target_length:
                pool_out = pool_out[:, :, :target_length]
            else:
                pad_size = target_length - pool_out.shape[2]
                pool_out = torch.nn.functional.pad(pool_out, (0, pad_size), mode='constant', value=0)
        
        # 拼接所有输出（现在所有输出的长度都一致了）
        x = torch.cat(conv_outputs + [pool_out], dim=1)
        x = self.bn(x)
        x = self.activation(x)
        
        return x


class ShortcutLayer(nn.Module):
    """残差连接层"""
    
    def __init__(self, in_channels, out_channels):
        super(ShortcutLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
    
    def forward(self, x, residual):
        shortcut = self.conv(x)
        shortcut = self.bn(shortcut)
        out = shortcut + residual
        out = self.activation(out)
        return out


class InceptionTime(nn.Module):
    """Inception Time模型"""
    
    def __init__(self, input_shape, n_classes, depth=6, kernel_size=40, 
                 bottleneck_size=32, nb_filters=32, use_residual=True):
        """
        Args:
            input_shape: 输入形状 (seq_len, n_channels) 例如 (1000, 12)
            n_classes: 类别数
            depth: 模型深度（Inception模块的数量）
            kernel_size: 卷积核大小
            bottleneck_size: bottleneck层大小
            nb_filters: 每个Inception模块的输出通道数
            use_residual: 是否使用残差连接
        """
        super(InceptionTime, self).__init__()
        
        seq_len, n_channels = input_shape
        
        self.use_residual = use_residual
        self.depth = depth
        
        # 输入层
        self.input_layer = nn.Identity()
        
        # Inception模块
        self.inception_modules = nn.ModuleList()
        current_channels = n_channels
        
        for d in range(depth):
            module = InceptionModule(
                in_channels=current_channels,
                out_channels=nb_filters,
                kernel_size=kernel_size,
                bottleneck_size=bottleneck_size,
                activation='relu'
            )
            self.inception_modules.append(module)
            current_channels = nb_filters * 4  # 4个分支的输出拼接
        
        # 残差连接层
        if use_residual:
            self.shortcut_layers = nn.ModuleList()
            residual_input_channels = n_channels  # 第一次残差连接的输入通道数
            for d in range(depth):
                if d % 3 == 2:  # 每3个模块添加一次残差连接
                    shortcut = ShortcutLayer(residual_input_channels, current_channels)
                    self.shortcut_layers.append(shortcut)
                    residual_input_channels = current_channels  # 后续残差连接的输入通道数
                else:
                    self.shortcut_layers.append(None)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # 输出层
        self.fc = nn.Linear(current_channels, n_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch, seq_len, n_channels)
        # 转换为 (batch, n_channels, seq_len)
        x = x.transpose(1, 2)
        
        input_res = x
        
        # 通过Inception模块
        for d in range(self.depth):
            x = self.inception_modules[d](x)
            
            # 残差连接（每3个模块）
            if self.use_residual and d % 3 == 2:
                if self.shortcut_layers[d] is not None:
                    x = self.shortcut_layers[d](input_res, x)
                    input_res = x
        
        # Global Average Pooling
        x = self.gap(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)
        
        # 输出层
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x


def create_inception_time_model(input_shape, n_classes, depth=9, kernel_size=60,
                                bottleneck_size=32, nb_filters=32, use_residual=True):
    """
    创建Inception Time模型
    
    Args:
        input_shape: 输入形状 (seq_len, n_channels)
        n_classes: 类别数
        depth: 模型深度
        kernel_size: 卷积核大小
        bottleneck_size: bottleneck层大小
        nb_filters: 每个Inception模块的输出通道数
        use_residual: 是否使用残差连接
    
    Returns:
        model: InceptionTime模型
    """
    model = InceptionTime(
        input_shape=input_shape,
        n_classes=n_classes,
        depth=depth,
        kernel_size=kernel_size,
        bottleneck_size=bottleneck_size,
        nb_filters=nb_filters,
        use_residual=use_residual
    )
    return model

