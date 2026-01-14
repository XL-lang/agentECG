"""
Inception Time分类器 - 用于推理
"""
import os
import pickle
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from .inception_time import create_inception_time_model
from data.EcgSignals import EcgSignals
# 使用标准的 os.path 路径处理（替代 pathlib.Path）
current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = two levels up from current_dir
project_root = os.path.dirname(os.path.dirname(current_dir))
project_root = os.path.join(project_root, 'models', 'pytorch_inception')
output_dir = os.path.join(project_root, 'output')
checkpoint_path = os.path.join(
    output_dir, 'checkpoints', 'best_checkpoint.pth')
data_dir = os.path.join(output_dir, 'data')
                   

class ECGSignalClassifier:    
    def __init__(self, ecg_signals: EcgSignals):
        self.ecg_signals = ecg_signals                                          
        self.classifier = InceptionClassifier(checkpoint_path, data_dir)
        self.default_fs = 500

    def trans_fs(self):
        if self.ecg_signals.fs != self.default_fs:
            self.ecg_signals.trans_fs(self.default_fs)

    def predict(self, return_all=False):
        signals = self.ecg_signals.signals
        if not isinstance(signals, np.ndarray):
            raise ValueError(f"expected numpy array, but got {type(signals)}")
        predictions = self.classifier.predict(signals, return_all=return_all)
        return predictions

    def report_predict_result(self, report_rate=0.1):
        predictions = self.predict(return_all=True)
        total_num = len(predictions)
        report_num = int(total_num * report_rate)
        report = f"The report format is diagnostic abbreviation plus confidence. The report is as follows:\n"
        predictions_list = list(predictions.items())
        predictions_list.sort(key=lambda x: x[1], reverse=True)
        for class_name, confidence in predictions_list[:report_num]:
            report += f"{class_name}: {confidence:.4f}\n"
        return report


class InceptionClassifier:
    """Inception Time分类器，用于ECG信号分类"""

    def __init__(self, checkpoint_path, data_dir=None, device=None):
        """
        初始化分类器

        Args:
            checkpoint_path: 模型checkpoint路径（.pth文件）
            data_dir: 数据目录路径（包含scaler和mlb的目录，默认从checkpoint同目录的../data/查找）
            device: 设备（'cuda'或'cpu'，默认自动检测）
        """
        self.checkpoint_path = checkpoint_path
        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)

        # 确定数据目录
        if data_dir is None:
            # 默认从checkpoint目录的../data/查找
            checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
            data_dir = os.path.join(os.path.dirname(checkpoint_dir), 'data')
        self.data_dir = data_dir

        # 加载模型和预处理器
        self._load_model()
        self._load_preprocessors()

    def _load_model(self):
        """加载模型"""
        print(f"加载模型: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # 获取模型参数
        if 'input_shape' in checkpoint:
            self.input_shape = checkpoint['input_shape']
            self.depth = checkpoint.get('depth', 9)
            self.kernel_size = checkpoint.get('kernel_size', 60)
            self.bottleneck_size = checkpoint.get('bottleneck_size', 32)
            self.nb_filters = checkpoint.get('nb_filters', 32)
            self.n_classes = checkpoint.get('n_classes', None)
        else:
            # 如果没有保存，使用默认值（对应all任务的配置）
            self.input_shape = (1000, 12)  # (seq_len, n_channels)
            self.depth = 9
            self.kernel_size = 60
            self.bottleneck_size = 32
            self.nb_filters = 32
            self.n_classes = None

        # 创建模型
        if self.n_classes is None:
            # 如果不知道类别数，从mlb加载
            mlb_path = os.path.join(self.data_dir, 'mlb.pkl')
            if os.path.exists(mlb_path):
                with open(mlb_path, 'rb') as f:
                    mlb = pickle.load(f)
                self.n_classes = len(mlb.classes_)
            else:
                raise ValueError("无法确定类别数，请确保data_dir包含mlb.pkl文件")

        self.model = create_inception_time_model(
            input_shape=self.input_shape,
            n_classes=self.n_classes,
            depth=self.depth,
            kernel_size=self.kernel_size,
            bottleneck_size=self.bottleneck_size,
            nb_filters=self.nb_filters,
            use_residual=True
        )

        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果checkpoint直接是state_dict
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        print(
            f"模型加载完成: input_shape={self.input_shape}, n_classes={self.n_classes}")

    def _load_preprocessors(self):
        """加载预处理器（scaler和mlb）"""
        # 加载StandardScaler
        scaler_path = os.path.join(self.data_dir, 'standard_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"加载StandardScaler: {scaler_path}")
        else:
            print(f"警告: 未找到scaler文件 {scaler_path}，将不使用标准化")
            self.scaler = None

        # 加载MultiLabelBinarizer
        mlb_path = os.path.join(self.data_dir, 'mlb.pkl')
        if os.path.exists(mlb_path):
            with open(mlb_path, 'rb') as f:
                self.mlb = pickle.load(f)
            self.class_names = list(self.mlb.classes_)
            print(f"加载MultiLabelBinarizer: {mlb_path}")
            print(f"类别数: {len(self.class_names)}")
        else:
            raise FileNotFoundError(f"未找到mlb文件: {mlb_path}")

    def _preprocess_signal(self, ecg_signal):
        """
        预处理ECG信号

        Args:
            ecg_signal: ECG信号，形状为 (:, 12) 或 (N, :, 12) 的numpy array

        Returns:
            预处理后的信号，形状为 (N, seq_len, 12)
        """
        ecg_signal = np.array(ecg_signal, dtype=np.float32)

        # 处理输入形状
        if ecg_signal.ndim == 2:
            # 输入形状为 (:, 12)，需要添加batch维度
            if ecg_signal.shape[1] != 12:
                raise ValueError(f"ECG信号应该有12个导联，但得到 {ecg_signal.shape[1]} 个")
            ecg_signal = ecg_signal[np.newaxis, :, :]  # (1, :, 12)
        elif ecg_signal.ndim == 3:
            # 输入形状为 (N, :, 12)
            if ecg_signal.shape[2] != 12:
                raise ValueError(f"ECG信号应该有12个导联，但得到 {ecg_signal.shape[2]} 个")
        else:
            raise ValueError(
                f"ECG信号应该是2D (:, 12) 或3D (N, :, 12)，但得到 {ecg_signal.ndim}D")

        seq_len, n_channels = self.input_shape
        n_samples = ecg_signal.shape[0]
        current_length = ecg_signal.shape[1]

        # 处理长度：如果长度不匹配，进行裁剪或填充
        processed_signals = []
        for i in range(n_samples):
            signal = ecg_signal[i]  # (current_length, 12)

            if current_length != seq_len:
                if current_length > seq_len:
                    # 裁剪到目标长度（取中间部分）
                    start_idx = (current_length - seq_len) // 2
                    signal = signal[start_idx:start_idx + seq_len]
                else:
                    # 填充到目标长度（在两端填充零）
                    pad_size = seq_len - current_length
                    pad_before = pad_size // 2
                    pad_after = pad_size - pad_before
                    signal = np.pad(
                        signal, ((pad_before, pad_after), (0, 0)), mode='constant', constant_values=0)

            processed_signals.append(signal)

        processed_signals = np.array(processed_signals)  # (N, seq_len, 12)

        # 标准化
        if self.scaler is not None:
            processed_signals_scaled = []
            for signal in processed_signals:
                signal_shape = signal.shape
                signal_scaled = self.scaler.transform(
                    signal.flatten()[:, np.newaxis]).reshape(signal_shape)
                processed_signals_scaled.append(signal_scaled)
            processed_signals = np.array(processed_signals_scaled)

        return processed_signals

    def predict(self, ecg_signal, return_all=False):
        """
        对ECG信号进行分类预测

        Args:
            ecg_signal: ECG信号，形状为 (:, 12) 或 (N, :, 12) 的numpy array
            return_all: 如果True，返回所有类别的置信度；如果False，只返回置信度>0.5的类别

        Returns:
            如果输入是单个样本 (:, 12)，返回一个字典 {"class_name": confidence}
            如果输入是多个样本 (N, :, 12)，返回一个列表，每个元素是一个字典
        """
        # 预处理
        processed_signals = self._preprocess_signal(ecg_signal)

        # 转换为tensor
        X = torch.FloatTensor(processed_signals).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(X)
            predictions = outputs.cpu().numpy()  # (N, n_classes)

        # 转换为字典格式
        results = []
        for pred in predictions:
            result_dict = {}
            for i, class_name in enumerate(self.class_names):
                confidence = float(pred[i])
                if return_all or confidence > 0.5:
                    result_dict[class_name] = confidence

            # 按置信度排序
            result_dict = dict(
                sorted(result_dict.items(), key=lambda x: x[1], reverse=True))
            results.append(result_dict)

        # 如果输入是单个样本，返回单个字典；否则返回列表
        if ecg_signal.ndim == 2:
            return results[0]
        else:
            return results

    def predict_proba(self, ecg_signal):
        """
        返回所有类别的概率（置信度）

        Args:
            ecg_signal: ECG信号，形状为 (:, 12) 或 (N, :, 12) 的numpy array

        Returns:
            如果输入是单个样本，返回一个字典 {"class_name": confidence}
            如果输入是多个样本，返回一个列表，每个元素是一个字典
        """
        return self.predict(ecg_signal, return_all=True)


# 便捷函数
def load_classifier(checkpoint_path=checkpoint_path, data_dir=data_dir, device=None):
    """
    加载分类器的便捷函数

    Args:
        checkpoint_path: 模型checkpoint路径
        data_dir: 数据目录路径
        device: 设备

    Returns:
        InceptionClassifier实例
    """
    return InceptionClassifier(checkpoint_path, data_dir, device)


if __name__ == '__main__':

    # 加载分类器
    classifier = load_classifier()

    # 创建示例ECG信号（随机数据，仅用于演示）
    print("\n测试分类器...")
    import wfdb
    path = "/home/xl/agentECG/dataset/dataset_1/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/01000/01744_hr"
    siganls = wfdb.rdrecord(path).p_signal[:1000, :12]  # 取前1000个采样点，12导联
    # 预测
    result = classifier.predict(siganls, return_all=True)

    print(f"\n预测结果:")
    for i, (class_name, confidence) in enumerate(list(result.items())[:]):
        print(f"  {class_name}: {confidence:.4f}")

    # 获取所有类别的概率
    all_probs = classifier.predict_proba(siganls)
    print(f"\n总类别数: {len(all_probs)}")
    print(f"置信度>0.5的类别数: {len([v for v in all_probs.values() if v > 0.5])}")
