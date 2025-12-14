import copy
import numpy as np
from typing import  Optional
from numpy.typing import NDArray
from models.ecgseg.ecgseg import  extract_ecg_wave_segments

class EcgSignals:
    def __init__(self, signals, fields):
        self.original_signals = signals.copy()
        self.original_fields = copy.deepcopy(fields)
        self.signals = signals
        self.fields = fields
        self.leads = fields.get('sig_name', [])
        self.fs = fields.get('fs', None)
        self.wave_types = ['P wave', 'QRS complex', 'T wave']
        self.interval_types = ['PR interval', 'QT interval', 'ST segment']
    

    def get_fs(self)-> int:
        return self.fs
    def base_info(self) -> dict:
        """
        获取心电图信号的基本信息。

        返回:
            dict: 包含以下键的字典:
                - 'n_leads': 导联数量
                - 'n_samples': 每个导联的采样点数量
                - 'fs': 采样频率 (Hz)
                - 'leads': 导联名称列表
        """
        n_leads = self.signals.shape[1] if self.signals is not None else 0
        n_samples = self.signals.shape[0] if self.signals is not None else 0
        return {
            'n_leads': n_leads,
            'n_samples': n_samples,
            'fs': self.fs,
            'leads': self.leads,
        }
    
    def get_lead_signals(self, lead_name: Optional[str] = None) -> (NDArray[np.float64]):
        """
        获取指定导联的信号数据。

        参数:
            lead_name (str, optional): 要获取的导联名称。
                                     如果为 None，返回所有导联的信号。
        
        返回:
            NDArray[np.float64]: 指定导联或所有导联的信号数据。
                - 如果 lead_name 为 None: 形状为 (n_samples, n_leads) - 所有导联
                - 如果指定 lead_name: 形状为 (n_samples,) - 单个导联
        
        异常:
            ValueError: 当指定的导联名称不存在时抛出。
        """
        if lead_name is None:
            return self.signals  # shape: (n_samples, n_leads)
            
        if lead_name not in self.leads:
            raise ValueError(f"Lead name '{lead_name}' not found in available leads: {self.leads}")
        
        lead_index = self.leads.index(lead_name)
        return self.signals[:, lead_index]  # shape: (n_samples,)
    
    def get_lead_segment(self, lead_name:str , segment_name:str):
        """
        查询指定导联的波或间期片段。

        支持查询波类型（如 'P wave', 'QRS complex', 'T wave'）和间期类型（如 'PR interval', 'QT interval', 'ST segment'）。

        参数:
            lead_name (str): 要提取片段的导联名称。
            segment_name (str): 要查询的片段名称，可为波类型或间期类型。

        返回:
            List[Tuple[int, int]]: 指定片段的 (开始索引, 结束索引) 元组列表。

        异常:
            ValueError: 当 segment_name 不在可用片段类型中时抛出。
        """
        lead_signal = self.get_lead_signals(lead_name)
        segments = extract_ecg_wave_segments(lead_signal)
        for wave_type in  self.wave_types:
                if wave_type not in segments:
                    raise ValueError(f"Wave type '{wave_type}' not found in segments. Available types: {list(segments.keys())}")
        matched_segments = self.interpolate_intervals(segments)
        for interval_type in  self.interval_types:
            if interval_type not in matched_segments:
                raise ValueError(f"Interval type '{interval_type}' not found in matched segments. Available types: {list(matched_segments.keys())}")
            segments[interval_type] = matched_segments[interval_type]
        if segment_name not in segments:
            raise ValueError(f"Segment name '{segment_name}' not found in segments. Available types: {list(segments.keys())}")
        return segments[segment_name]
    def interpolate_intervals(self, segments: dict):
        """
        匹配 P 波、QRS 波群和 T 波片段，计算心电图间期。
        基于间期长度去除离群值（40% 阈值）。
        
        匹配策略:
            - 以 QRS 波群为锚点
            - 对每个 QRS 波群，找到其前最近的 P 波（P 波结束 <= QRS 开始）
            - 对每个 QRS 波群，找到其后最近的 T 波（T 波开始 >= QRS 结束）
            - 允许一个波形被多次匹配（处理心律不齐等情况）
        
        参数:
            segments (dict): 包含 'P wave'、'QRS complex'、'T wave' 键的字典，
                           每个键对应一个 (开始索引, 结束索引) 元组的列表。
        
        返回:
            dict: 包含以下键的字典:
                - 'PR interval': (开始, 结束) 元组列表，从 P 波开始到 QRS 开始
                - 'QT interval': (开始, 结束) 元组列表，从 QRS 开始到 T 波结束
                - 'ST segment': (开始, 结束) 元组列表，从 QRS 结束到 T 波开始
        
        离群值检测:
            使用基于中位数绝对偏差(MAD)的改进 Z 分数方法，
            过滤掉长度偏离中位数超过 40% 的间期。
        """
        p_waves = segments.get('P wave', [])
        qrs_complexes = segments.get('QRS complex', [])
        t_waves = segments.get('T wave', [])
        
        # Match segments based on temporal proximity
        matched_beats = self._match_waves(p_waves, qrs_complexes, t_waves)
        
        # Calculate intervals
        pr_intervals = []
        qt_intervals = []
        st_segments = []
        
        for p, qrs, t in matched_beats:
            if p is not None and qrs is not None:
                pr_intervals.append((p[0], qrs[0]))  # P start to QRS start
            
            if qrs is not None and t is not None:
                qt_intervals.append((qrs[0], t[1]))  # QRS start to T end
                st_segments.append((qrs[1], t[0]))   # QRS end to T start
        
        # Remove outliers (40% threshold)
        pr_intervals = self._remove_outliers(pr_intervals, threshold=0.4)
        qt_intervals = self._remove_outliers(qt_intervals, threshold=0.4)
        st_segments = self._remove_outliers(st_segments, threshold=0.4)
        
        return {
            'PR interval': pr_intervals,
            'QT interval': qt_intervals,
            'ST segment': st_segments,
        }
    
    def _match_waves(self, p_waves, qrs_complexes, t_waves):
        """
        Match P wave, QRS complex, and T wave based on temporal proximity.
        Allows one wave to be matched multiple times.
        
        Strategy: For each QRS complex, find the nearest P wave before it and nearest T wave after it.
        
        Returns:
            List of tuples (p, qrs, t) where each element is a (start, end) tuple or None
        """
        matched_beats = []
        
        # For each QRS complex (anchor point)
        for qrs in qrs_complexes:
            qrs_start = qrs[0]
            qrs_end = qrs[1]
            
            # Find the closest P wave before this QRS (P wave ends before QRS starts)
            p_wave = None
            min_p_distance = float('inf')
            for p in p_waves:
                if p[1] <= qrs_start:  # P wave ends before QRS starts
                    distance = qrs_start - p[1]
                    if distance < min_p_distance:
                        min_p_distance = distance
                        p_wave = p
            
            # Find the closest T wave after this QRS (T wave starts after QRS ends)
            t_wave = None
            min_t_distance = float('inf')
            for t in t_waves:
                if t[0] >= qrs_end:  # T wave starts after QRS ends
                    distance = t[0] - qrs_end
                    if distance < min_t_distance:
                        min_t_distance = distance
                        t_wave = t
            
            matched_beats.append((p_wave, qrs, t_wave))
        
        return matched_beats
    
    def _remove_outliers(self, intervals, threshold=0.4):
        """
        Remove outliers from intervals based on interval length.
        
        Args:
            intervals: List of (start, end) tuples
            threshold: Percentage threshold for outlier detection (default 0.4 = 40%)
        
        Returns:
            List of intervals with outliers removed
        """
        if len(intervals) < 3:
            return intervals
        
        # Calculate interval lengths
        lengths = [end - start for start, end in intervals]
        
        # Calculate median and median absolute deviation
        median_length = np.median(lengths)
        mad = np.median(np.abs(np.array(lengths) - median_length))
        
        # Use MAD-based outlier detection (more robust than standard deviation)
        # Modified Z-score using MAD
        if mad == 0:
            # If MAD is 0, use standard deviation instead
            std_length = np.std(lengths)
            if std_length == 0:
                return intervals
            lower_bound = median_length * (1 - threshold)
            upper_bound = median_length * (1 + threshold)
        else:
            # Use 3 * MAD as threshold (equivalent to ~3 sigma for normal distribution)
            modified_z_scores = 0.6745 * np.abs(np.array(lengths) - median_length) / mad
            # Keep intervals where modified z-score < 3.5 (common threshold)
            filtered_intervals = [intervals[i] for i in range(len(intervals)) 
                                if modified_z_scores[i] < 3.5]
            
            # Additional check: remove if length differs by more than threshold from median
            lower_bound = median_length * (1 - threshold)
            upper_bound = median_length * (1 + threshold)
            filtered_intervals = [interval for interval in filtered_intervals 
                                if lower_bound <= (interval[1] - interval[0]) <= upper_bound]
            
            return filtered_intervals
        
        # Filter intervals within bounds
        return [interval for interval in intervals 
                if lower_bound <= (interval[1] - interval[0]) <= upper_bound]

        
    
        
    def __str__(self):
        """Return a user-friendly string representation"""
        return f"EcgSignals(signals_shape={getattr(self.signals, 'shape', 'unknown')}, fields={self.fields})"
    
    def __repr__(self):
        """Return a developer-friendly string representation"""
        return f"EcgSignals(signals={type(self.signals).__name__}, fields={self.fields})"