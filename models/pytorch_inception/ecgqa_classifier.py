"""
Inception Time分类器 - 用于推理
支持按template_id自动加载模型
"""
import os
import json
import pickle
from typing import List, Optional
import numpy as np
import pandas as pd
import ast
import json
import torch
from sklearn.preprocessing import StandardScaler
from .inception_time import create_inception_time_model
from data.EcgSignals import EcgSignals
from dataset.config import ALLOWED_QUERY_TEMPLATE_ID_SET
import wfdb
# 使用标准的 os.path 路径处理
current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = two levels up from current_dir
project_root = os.path.dirname(os.path.dirname(current_dir))
output_base_dir = os.path.join(project_root, 'output', 'templates')


class InceptionClassifier:
    """Inception Time分类器，用于ECG信号分类，支持按template_id自动加载模型"""

    def __init__(self, template_id: int, output_dir: Optional[str] = None, device: Optional[str] = None, use_best: bool = False):
        """
        初始化分类器

        Args:
            template_id: template ID，用于自动加载对应的模型
            output_dir: 输出目录的基础路径（默认使用项目根目录下的output/templates）
            device: 设备（'cuda'或'cpu'，默认自动检测）
            use_best: 如果True，使用best_checkpoint.pth；如果False，优先使用final_model.pth
        """
        self.template_id = template_id
        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)

        # 确定输出目录
        if output_dir is None:
            self.output_dir = output_base_dir
        else:
            self.output_dir = output_dir

        # 构建路径
        self.template_output_dir = os.path.join(
            self.output_dir, f'template_{template_id}')
        self.checkpoint_dir = os.path.join(
            self.template_output_dir, 'checkpoints')
        self.data_dir = os.path.join(self.template_output_dir, 'data')

        # 检查目录是否存在
        if not os.path.exists(self.template_output_dir):
            raise FileNotFoundError(
                f"Template {template_id} 的输出目录不存在: {self.template_output_dir}")

        # 加载模型和预处理器
        self._load_class_info()
        self._load_model(use_best=use_best)
        self._load_preprocessors()

    def _load_class_info(self):
        """加载类别信息"""
        class_info_path = os.path.join(self.data_dir, 'class_info.json')
        if not os.path.exists(class_info_path):
            raise FileNotFoundError(
                f"未找到class_info.json文件: {class_info_path}")

        with open(class_info_path, 'r') as f:
            self.class_info = json.load(f)

        self.template_id = self.class_info['template_id']
        self.input_ecg_num = self.class_info['input_ecg_num']
        self.output_dim = self.class_info['output_dim']
        self.output_classes = sorted(self.class_info['output_classes'])
        self.classification_info = self.class_info.get(
            'classification_info', {})

    def _load_model(self, use_best=False):
        """加载模型"""
        # 优先使用final_model.pth（包含完整的模型架构参数）
        final_model_path = os.path.join(
            self.checkpoint_dir, f'template_{self.template_id}_final_model.pth')
        best_checkpoint_path = os.path.join(
            self.checkpoint_dir, f'template_{self.template_id}_best_checkpoint.pth')

        if use_best and os.path.exists(best_checkpoint_path):
            checkpoint_path = best_checkpoint_path
        elif os.path.exists(final_model_path):
            checkpoint_path = final_model_path

        elif os.path.exists(best_checkpoint_path):
            checkpoint_path = best_checkpoint_path

        else:
            raise FileNotFoundError(
                f"未找到模型文件。请检查以下路径:\n"
                f"  {final_model_path}\n"
                f"  {best_checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 获取模型参数（优先从checkpoint读取，否则使用默认值）
        if 'input_shape' in checkpoint:
            self.input_shape = checkpoint['input_shape']
        else:
            # 默认值（对应train_ecgqa.py中的seq_len=1000）
            self.input_shape = (1000, 12)  # (seq_len, n_channels)

        self.depth = checkpoint.get('depth', 9)
        self.kernel_size = checkpoint.get('kernel_size', 60)
        self.bottleneck_size = checkpoint.get('bottleneck_size', 32)
        self.nb_filters = checkpoint.get('nb_filters', 32)
        self.n_classes = checkpoint.get('n_classes', self.output_dim)

        # 创建模型
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

    def _load_preprocessors(self):
        """加载预处理器（scaler）"""
        # 加载StandardScaler
        scaler_path = os.path.join(self.data_dir, 'standard_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            print(f"警告: 未找到scaler文件 {scaler_path}，将不使用标准化")
            self.scaler = None

    def _preprocess_signal(self, ecg_signal):
        """
        预处理ECG信号（支持多ECG输入）

        处理逻辑与train_ecgqa.py中的ECGQAClassificationDatasetWrapper._load_sample一致

        Args:
            ecg_signal: ECG信号
                - 如果input_ecg_num == 1: 单个ECG，形状为 (:, 12) 的numpy array 或列表 [ecg]
                - 如果input_ecg_num > 1: ECG列表 [ecg1, ecg2, ..., ecgN]，其中N = input_ecg_num

        Returns:
            预处理后的信号，形状为 (1, seq_len, 12) - 单个样本的batch
        """
        seq_len, n_channels = self.input_shape

        # 处理输入：如果是单个ECG信号，转换为列表格式
        if isinstance(ecg_signal, np.ndarray):
            if ecg_signal.ndim == 2:
                # 单个样本 (:, 12)
                ecg_signals = [ecg_signal]
            elif ecg_signal.ndim == 3:
                # 多个样本 (N, :, 12)
                ecg_signals = [ecg_signal[i]
                               for i in range(ecg_signal.shape[0])]
            else:
                raise ValueError(
                    f"ECG信号应该是2D (:, 12) 或3D (N, :, 12)，但得到 {ecg_signal.ndim}D")
        elif isinstance(ecg_signal, list):
            # 列表格式（多ECG输入）
            ecg_signals = ecg_signal
        else:
            raise ValueError(f"不支持的输入类型: {type(ecg_signal)}")

        # 根据input_ecg_num处理输入
        if self.input_ecg_num > 1:
            # 多ECG输入：需要确保有足够数量的ECG
            if len(ecg_signals) < self.input_ecg_num:
                # 如果ECG数量不足，重复最后一个
                while len(ecg_signals) < self.input_ecg_num:
                    ecg_signals.append(ecg_signals[-1])
            elif len(ecg_signals) > self.input_ecg_num:
                # 如果ECG数量过多，截取前input_ecg_num个
                ecg_signals = ecg_signals[:self.input_ecg_num]
        else:
            # 单ECG输入：只使用第一个
            ecg_signals = [ecg_signals[0]]

        # 处理每个ECG信号：调整长度
        processed_signals = []
        for signal in ecg_signals:
            signal = np.array(signal, dtype=np.float32)

            # 确保是12导联
            if signal.shape[1] != 12:
                raise ValueError(f"ECG信号应该有12个导联，但得到 {signal.shape[1]} 个")

            # 调整长度到seq_len
            current_length = signal.shape[0]
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
                        signal, ((pad_before, pad_after), (0, 0)),
                        mode='constant', constant_values=0
                    )

            processed_signals.append(signal)

        # 如果有多个ECG信号，拼接后下采样
        if self.input_ecg_num > 1:
            combined_signal = np.concatenate(processed_signals, axis=0)
            if combined_signal.shape[0] > seq_len:
                step = combined_signal.shape[0] // seq_len
                combined_signal = combined_signal[::step][:seq_len]
            elif combined_signal.shape[0] < seq_len:
                pad_size = seq_len - combined_signal.shape[0]
                pad_before = pad_size // 2
                pad_after = pad_size - pad_before
                combined_signal = np.pad(
                    combined_signal, ((pad_before, pad_after), (0, 0)),
                    mode='constant', constant_values=0
                )
            final_signal = combined_signal
        else:
            final_signal = processed_signals[0]

        # 标准化
        if self.scaler is not None:
            signal_shape = final_signal.shape  # (seq_len, 12)
            final_signal_reshaped = final_signal.reshape(-1, 12)
            final_signal_scaled = self.scaler.transform(final_signal_reshaped)
            final_signal = final_signal_scaled.reshape(signal_shape)

        # 添加batch维度
        final_signal = final_signal[np.newaxis, :, :]  # (1, seq_len, 12)

        return final_signal

    def predict(self, ecg_signal, return_all=False):
        """
        对ECG信号进行分类预测

        Args:
            ecg_signal: ECG信号列表，格式为 list[array]，每个array形状为 (:, 12)
                - 输入的ECG数量必须是 self.input_ecg_num 的自然数倍
                - 倍数表示样本数量，例如：
                    - input_ecg_num=1: [ecg1] 表示1个样本，[ecg1, ecg2] 表示2个样本
                    - input_ecg_num=2: [ecg1, ecg2] 表示1个样本，[ecg1, ecg2, ecg3, ecg4] 表示2个样本
            return_all: 如果True，返回所有类别的置信度；如果False，只返回置信度>0.5的类别

        Returns:
            列表，每个元素是一个字典 {"class_name": confidence}，按置信度降序排列
            列表长度等于输入样本数量
        """
        # 验证输入格式
        if not isinstance(ecg_signal, list):
            raise ValueError(
                f"输入必须是列表格式 list[array]，但得到 {type(ecg_signal)}")

        if len(ecg_signal) == 0:
            raise ValueError("输入列表不能为空")

        # 验证每个ECG的形状
        for i, ecg in enumerate(ecg_signal):
            ecg = np.array(ecg)
            if ecg.ndim != 2 or ecg.shape[1] != 12:
                raise ValueError(
                    f"第 {i} 个ECG信号形状不正确，应为 (:, 12)，但得到 {ecg.shape}")

        # 验证ECG数量是否是input_ecg_num的倍数
        n_ecgs = len(ecg_signal)
        if n_ecgs % self.input_ecg_num != 0:
            raise ValueError(
                f"Template {self.template_id} 需要 {self.input_ecg_num} 个ECG信号组成一个样本，"
                f"但提供了 {n_ecgs} 个ECG（无法整除）。")

        # 按input_ecg_num分组，每组是一个样本
        n_samples = n_ecgs // self.input_ecg_num
        samples = []
        for i in range(n_samples):
            start_idx = i * self.input_ecg_num
            end_idx = start_idx + self.input_ecg_num
            samples.append(ecg_signal[start_idx:end_idx])

        # 预处理所有样本
        processed_signals = []
        for sample_ecgs in samples:
            # sample_ecgs是一个列表，包含该样本所需的所有ECG信号
            processed = self._preprocess_signal(sample_ecgs)
            processed_signals.append(processed)

        # 合并为batch
        X = np.vstack(processed_signals)  # (N, seq_len, 12)
        X = torch.FloatTensor(X).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(X)
            predictions = outputs.cpu().numpy()  # (N, n_classes)

        # 转换为字典格式
        results = []
        for pred in predictions:
            result_dict = {}
            for i, class_name in enumerate(self.output_classes):
                confidence = float(pred[i])
                if return_all or confidence > 0.5:
                    result_dict[class_name] = confidence

            # 按置信度排序
            result_dict = dict(
                sorted(result_dict.items(), key=lambda x: x[1], reverse=True))
            results.append(result_dict)

        # 不管是不是单样本，都返回list[dict]
        return results

    def predict_proba(self, ecg_signal):
        """
        返回所有类别的概率（置信度）

        Args:
            ecg_signal: ECG信号列表，格式为 list[array]，每个array形状为 (:, 12)
                - 输入的ECG数量必须是 self.input_ecg_num 的自然数倍
                - 倍数表示样本数量

        Returns:
            列表，每个元素是一个字典 {"class_name": confidence}，包含所有类别的置信度，按置信度降序排列
            列表长度等于输入样本数量
        """
        return self.predict(ecg_signal, return_all=True)


# 便捷函数
def load_classifier(template_id: int, output_dir: Optional[str] = None, device: Optional[str] = None, use_best: bool = False):
    """
    加载分类器的便捷函数

    Args:
        template_id: template ID
        output_dir: 输出目录的基础路径
        device: 设备
        use_best: 如果True，使用best_checkpoint.pth；如果False，优先使用final_model.pth

    Returns:
        InceptionClassifier实例
    """
    return InceptionClassifier(template_id=template_id, output_dir=output_dir, device=device, use_best=use_best)


class ECGQAClassifierManager:
    def __init__(self, ecgs: List[EcgSignals], seq_len: int = 1000):
        self.seq_len = seq_len
        self.load_basic_info(ecgs)

    def load_basic_info(self, ecgs: List[EcgSignals]):
        query_ids_path = "test_tools/ecgqa_info/ecgnum_ids_full.json"
        question_path = "test_tools/ecgqa_info/question.json"
        with open(query_ids_path, "r") as f:
            query_ids = json.load(f)[str(len(ecgs))]
        self.query_ids = [
            template_id for template_id in query_ids
            if template_id in ALLOWED_QUERY_TEMPLATE_ID_SET
        ]
        with open(question_path, "r") as f:
            questions = json.load(f)
        questions = {int(k): v for k, v in questions.items()}
        self.question = {}
        for id in self.query_ids:
            self.question[id] = questions[id]
        # 统一转换为列表格式 list[array]
        self.ecg = [ecg.signals[:self.seq_len, :] for ecg in ecgs]

    def report_result(self, template_id: int):
        predict_result = self.predict(template_id)
        n_samples = len(predict_result)
        result = f"The model processed {n_samples} sample(s). The following list contains the results for each sample, where each element is a list of tuples representing the results deemed credible by the model and their confidence levels: {predict_result}"
        return result

    def predict(self, template_id: int, show_all: bool = False):
        if template_id not in self.question:
            raise ValueError(
                f"Template {template_id} not found in questions, supported model dict is: {self.question}")

        classifier = load_classifier(template_id=template_id)
        proba_result = classifier.predict_proba(self.ecg)
        if show_all:
            print(proba_result)

        # predict_proba总是返回列表，处理所有样本
        if not isinstance(proba_result, list) or len(proba_result) == 0:
            raise ValueError(
                f"predict_proba返回了意外的类型或空列表: {type(proba_result)}")

        # 处理每个样本的结果
        all_results = []
        for proba_dict in proba_result:
            nums_over_05 = sum([1 for v in proba_dict.values() if v >= 0.5])
            if nums_over_05 == 0:
                # 返回最大值对应的kv（元组格式）
                result = max(proba_dict.items(), key=lambda x: x[1])
                all_results.append([result])
            elif nums_over_05 == 1:
                proba_dict = {k: v for k, v in proba_dict.items() if v >= 0.5}
                all_results.append(list(proba_dict.items()))
            else:
                proba_dict = {k: v for k, v in proba_dict.items() if v >= 0.5}
                keys = list(proba_dict.keys())
                singles = ["yes", "no", "null"]
                tf_max = None
                tf_max_value = 0
                for single in singles:
                    if single in keys:
                        if proba_dict[single] > tf_max_value:
                            tf_max = single
                            tf_max_value = proba_dict[single]
                        del proba_dict[single]
                if len(proba_dict) == 0:
                    all_results.append([(tf_max, tf_max_value)])
                else:
                    all_results.append(list(proba_dict.items()))

        return all_results


if __name__ == "__main__":    # 创建示例ECG信号
    path = "/home/xl/agentECG/dataset/dataset_1/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/01000/01942_hr"
    record = wfdb.rdrecord(path)
    classifier = load_classifier(template_id=10)
    if record is not None and hasattr(record, 'p_signal'):
        p_signal = getattr(record, 'p_signal', None)
        if p_signal is not None:
            signals = p_signal[:1000, :12]  # 取前1000个采样点，12导联

            # 预测（输入统一为列表格式）
            result = classifier.predict([signals], return_all=True)

            # predict总是返回列表，取第一个结果
            if isinstance(result, list) and len(result) > 0:
                result_dict = result[0]
                print(f"\n预测结果:")
                for i, (class_name, confidence) in enumerate(list(result_dict.items())[:10]):
                    print(f"  {class_name}: {confidence:.4f}")

                # 获取所有类别的概率（输入统一为列表格式）
                all_probs = classifier.predict_proba([signals])
                if isinstance(all_probs, list) and len(all_probs) > 0:
                    all_probs_dict = all_probs[0]
                    print(f"\n总类别数: {len(all_probs_dict)}")
                    print(
                        f"置信度>0.5的类别数: {len([v for v in all_probs_dict.values() if v > 0.5])}")
                else:
                    print(
                        f"\n警告: predict_proba返回了意外的类型或空列表: {type(all_probs)}")
            else:
                print(f"\n警告: predict返回了意外的类型或空列表: {type(result)}")
        else:
            print("无法获取ECG信号数据")
    else:
        print("无法加载ECG信号")
