"""
PTB-XL数据集加载器
"""
import os
import pickle
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import wfdb
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


class PTBXLDataset(Dataset):
    """PTB-XL数据集"""
    
    def __init__(self, X, y):
        """
        Args:
            X: ECG信号数据，形状为 (N, 1000, 12)
            y: 标签，形状为 (N, n_classes)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_ptbxl_dataset(datafolder, sampling_rate=100):
    """
    加载PTB-XL数据集
    
    Args:
        datafolder: 数据集文件夹路径
        sampling_rate: 采样率 (100 或 500)
    
    Returns:
        data: ECG信号数据，形状为 (N, 1000, 12) 或 (N, 5000, 12)
        labels_df: 标签DataFrame
    """
    # 加载标签数据
    labels_df = pd.read_csv(os.path.join(datafolder, 'ptbxl_database.csv'), index_col='ecg_id')
    labels_df.scp_codes = labels_df.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # 加载信号数据
    if sampling_rate == 100:
        cache_file = os.path.join(datafolder, 'raw100.npy')
        if os.path.exists(cache_file):
            print(f"从缓存加载数据: {cache_file}")
            data = np.load(cache_file, allow_pickle=True)
        else:
            print("从WFDB文件加载数据...")
            data = []
            for filename in tqdm(labels_df.filename_lr):
                signal, _ = wfdb.rdsamp(os.path.join(datafolder, filename))
                data.append(signal)
            data = np.array(data)
            print(f"保存缓存到: {cache_file}")
            np.save(cache_file, data)
    elif sampling_rate == 500:
        cache_file = os.path.join(datafolder, 'raw500.npy')
        if os.path.exists(cache_file):
            print(f"从缓存加载数据: {cache_file}")
            data = np.load(cache_file, allow_pickle=True)
        else:
            print("从WFDB文件加载数据...")
            data = []
            for filename in tqdm(labels_df.filename_hr):
                signal, _ = wfdb.rdsamp(os.path.join(datafolder, filename))
                data.append(signal)
            data = np.array(data)
            print(f"保存缓存到: {cache_file}")
            np.save(cache_file, data)
    else:
        raise ValueError(f"不支持的采样率: {sampling_rate}")
    
    return data, labels_df


def compute_label_aggregations(labels_df, datafolder, task='all'):
    """
    计算标签聚合
    
    Args:
        labels_df: 标签DataFrame
        datafolder: 数据集文件夹路径
        task: 任务类型 ('all', 'diagnostic', 'subdiagnostic', 'superdiagnostic', 'form', 'rhythm')
    
    Returns:
        处理后的标签DataFrame
    """
    aggregation_df = pd.read_csv(os.path.join(datafolder, 'scp_statements.csv'), index_col=0)
    
    if task == 'all':
        labels_df['all_scp'] = labels_df.scp_codes.apply(lambda x: list(set(x.keys())))
    elif task == 'diagnostic':
        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))
        labels_df['diagnostic'] = labels_df.scp_codes.apply(aggregate_all_diagnostic)
        labels_df['diagnostic_len'] = labels_df.diagnostic.apply(lambda x: len(x))
    elif task == 'subdiagnostic':
        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))
        labels_df['subdiagnostic'] = labels_df.scp_codes.apply(aggregate_subdiagnostic)
        labels_df['subdiagnostic_len'] = labels_df.subdiagnostic.apply(lambda x: len(x))
    elif task == 'superdiagnostic':
        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))
        labels_df['superdiagnostic'] = labels_df.scp_codes.apply(aggregate_diagnostic)
        labels_df['superdiagnostic_len'] = labels_df.superdiagnostic.apply(lambda x: len(x))
    elif task == 'form':
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]
        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))
        labels_df['form'] = labels_df.scp_codes.apply(aggregate_form)
        labels_df['form_len'] = labels_df.form.apply(lambda x: len(x))
    elif task == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]
        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))
        labels_df['rhythm'] = labels_df.scp_codes.apply(aggregate_rhythm)
        labels_df['rhythm_len'] = labels_df.rhythm.apply(lambda x: len(x))
    
    return labels_df


def select_data(data, labels_df, task='all', min_samples=0):
    """
    选择相关数据并转换为one-hot编码
    
    Args:
        data: ECG信号数据
        labels_df: 标签DataFrame
        task: 任务类型
        min_samples: 最小样本数
    
    Returns:
        X: 筛选后的信号数据
        labels_df: 筛选后的标签DataFrame
        y: one-hot编码的标签
        mlb: MultiLabelBinarizer
    """
    mlb = MultiLabelBinarizer()
    
    if task == 'all':
        # 过滤
        counts = pd.Series(np.concatenate(labels_df.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        labels_df.all_scp = labels_df.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        labels_df['all_scp_len'] = labels_df.all_scp.apply(lambda x: len(x))
        # 选择
        mask = labels_df.all_scp_len > 0
        X = data[mask]
        labels_df = labels_df[mask]
        mlb.fit(labels_df.all_scp.values)
        y = mlb.transform(labels_df.all_scp.values)
    elif task == 'diagnostic':
        X = data[labels_df.diagnostic_len > 0]
        labels_df = labels_df[labels_df.diagnostic_len > 0]
        mlb.fit(labels_df.diagnostic.values)
        y = mlb.transform(labels_df.diagnostic.values)
    elif task == 'subdiagnostic':
        counts = pd.Series(np.concatenate(labels_df.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        labels_df.subdiagnostic = labels_df.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        labels_df['subdiagnostic_len'] = labels_df.subdiagnostic.apply(lambda x: len(x))
        mask = labels_df.subdiagnostic_len > 0
        X = data[mask]
        labels_df = labels_df[mask]
        mlb.fit(labels_df.subdiagnostic.values)
        y = mlb.transform(labels_df.subdiagnostic.values)
    elif task == 'superdiagnostic':
        counts = pd.Series(np.concatenate(labels_df.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        labels_df.superdiagnostic = labels_df.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        labels_df['superdiagnostic_len'] = labels_df.superdiagnostic.apply(lambda x: len(x))
        mask = labels_df.superdiagnostic_len > 0
        X = data[mask]
        labels_df = labels_df[mask]
        mlb.fit(labels_df.superdiagnostic.values)
        y = mlb.transform(labels_df.superdiagnostic.values)
    elif task == 'form':
        counts = pd.Series(np.concatenate(labels_df.form.values)).value_counts()
        counts = counts[counts > min_samples]
        labels_df.form = labels_df.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        labels_df['form_len'] = labels_df.form.apply(lambda x: len(x))
        mask = labels_df.form_len > 0
        X = data[mask]
        labels_df = labels_df[mask]
        mlb.fit(labels_df.form.values)
        y = mlb.transform(labels_df.form.values)
    elif task == 'rhythm':
        counts = pd.Series(np.concatenate(labels_df.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        labels_df.rhythm = labels_df.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        labels_df['rhythm_len'] = labels_df.rhythm.apply(lambda x: len(x))
        mask = labels_df.rhythm_len > 0
        X = data[mask]
        labels_df = labels_df[mask]
        mlb.fit(labels_df.rhythm.values)
        y = mlb.transform(labels_df.rhythm.values)
    else:
        raise ValueError(f"不支持的任务类型: {task}")
    
    print(f"类别数: {len(mlb.classes_)}")
    print(f"类别: {mlb.classes_}")
    
    return X, labels_df, y, mlb


def preprocess_signals(X_train, X_val, X_test, scaler_save_path=None):
    """
    预处理信号数据（标准化）
    
    Args:
        X_train: 训练集信号
        X_val: 验证集信号
        X_test: 测试集信号
        scaler_save_path: StandardScaler保存路径
    
    Returns:
        X_train_scaled: 标准化后的训练集
        X_val_scaled: 标准化后的验证集
        X_test_scaled: 标准化后的测试集
        scaler: StandardScaler对象
    """
    # 标准化数据（均值为0，方差为1）
    scaler = StandardScaler()
    scaler.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))
    
    # 保存scaler
    if scaler_save_path:
        with open(scaler_save_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    # 应用标准化
    def apply_scaler(X, scaler):
        X_scaled = []
        for x in X:
            x_shape = x.shape
            x_scaled = scaler.transform(x.flatten()[:, np.newaxis]).reshape(x_shape)
            X_scaled.append(x_scaled)
        return np.array(X_scaled)
    
    X_train_scaled = apply_scaler(X_train, scaler)
    X_val_scaled = apply_scaler(X_val, scaler)
    X_test_scaled = apply_scaler(X_test, scaler)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def prepare_data(datafolder, task='all', sampling_rate=100, train_fold=8, val_fold=9, test_fold=10, 
                 min_samples=0, output_dir=None):
    """
    准备训练、验证和测试数据
    
    Args:
        datafolder: 数据集文件夹路径
        task: 任务类型
        sampling_rate: 采样率
        train_fold: 训练集fold
        val_fold: 验证集fold
        test_fold: 测试集fold
        min_samples: 最小样本数
        output_dir: 输出目录（用于保存scaler和mlb）
    
    Returns:
        X_train, X_val, X_test: 训练、验证、测试集信号
        y_train, y_val, y_test: 训练、验证、测试集标签
        n_classes: 类别数
        mlb: MultiLabelBinarizer
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # 加载数据
    print("加载PTB-XL数据集...")
    data, labels_df = load_ptbxl_dataset(datafolder, sampling_rate)
    
    # 计算标签聚合
    print(f"计算标签聚合 (task={task})...")
    labels_df = compute_label_aggregations(labels_df, datafolder, task)
    
    # 选择数据并转换为one-hot
    print("选择数据并转换为one-hot编码...")
    data, labels_df, y, mlb = select_data(data, labels_df, task, min_samples)
    
    # 根据strat_fold分割数据
    print("分割数据集...")
    X_test = data[labels_df.strat_fold == test_fold]
    y_test = y[labels_df.strat_fold == test_fold]
    X_val = data[labels_df.strat_fold == val_fold]
    y_val = y[labels_df.strat_fold == val_fold]
    X_train = data[labels_df.strat_fold <= train_fold]
    y_train = y[labels_df.strat_fold <= train_fold]
    
    # 预处理信号
    print("预处理信号数据...")
    scaler_path = os.path.join(output_dir, 'standard_scaler.pkl') if output_dir else None
    X_train, X_val, X_test, scaler = preprocess_signals(X_train, X_val, X_test, scaler_path)
    
    # 保存mlb
    if output_dir:
        mlb_path = os.path.join(output_dir, 'mlb.pkl')
        with open(mlb_path, 'wb') as f:
            pickle.dump(mlb, f)
    
    n_classes = y_train.shape[1]
    
    print(f"\n数据准备完成:")
    print(f"  训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"  验证集: {X_val.shape}, 标签: {y_val.shape}")
    print(f"  测试集: {X_test.shape}, 标签: {y_test.shape}")
    print(f"  类别数: {n_classes}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, n_classes, mlb

