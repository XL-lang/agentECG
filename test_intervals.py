"""
Test script for interval calculation and outlier removal
"""
import sys
import numpy as np

# Direct implementation for testing without import
class EcgSignalsTest:
    def __init__(self):
        pass
    
    def interpolate_intervals(self, segments: dict):
        """
        Match P wave, QRS complex, and T wave segments to calculate intervals.
        Removes outliers based on interval length (40% threshold).
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
            'matched_beats': matched_beats
        }
    
    def _match_waves(self, p_waves, qrs_complexes, t_waves):
        """
        Match P wave, QRS complex, and T wave based on temporal proximity.
        Strategy: For each QRS complex, find the nearest P wave before it and nearest T wave after it.
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
        """Remove outliers from intervals based on interval length."""
        if len(intervals) < 3:
            return intervals
        
        lengths = [end - start for start, end in intervals]
        median_length = np.median(lengths)
        mad = np.median(np.abs(np.array(lengths) - median_length))
        
        if mad == 0:
            std_length = np.std(lengths)
            if std_length == 0:
                return intervals
            lower_bound = median_length * (1 - threshold)
            upper_bound = median_length * (1 + threshold)
        else:
            modified_z_scores = 0.6745 * np.abs(np.array(lengths) - median_length) / mad
            filtered_intervals = [intervals[i] for i in range(len(intervals)) 
                                if modified_z_scores[i] < 3.5]
            
            lower_bound = median_length * (1 - threshold)
            upper_bound = median_length * (1 + threshold)
            filtered_intervals = [interval for interval in filtered_intervals 
                                if lower_bound <= (interval[1] - interval[0]) <= upper_bound]
            
            return filtered_intervals
        
        return [interval for interval in intervals 
                if lower_bound <= (interval[1] - interval[0]) <= upper_bound]

ecg = EcgSignalsTest()

# Create test data
test_segments = {
    'P wave': [
        (37, 80),
        (1044, 1146),
        (2328, 2431),
        (3643, 3747),
        (4969, 5000)
    ],
    'QRS complex': [
        (402, 508),
        (1665, 1769),
        (3000, 3102),
        (4324, 4429)
    ],
    'T wave': [
        (402, 508),
        (1665, 1769),
        (3000, 3102),
        (4324, 4429),
        (4969, 5000)
    ]
}



# Test the interpolate_intervals method
print("Testing interpolate_intervals method...")
print("\nInput segments:")
for wave_type, segments in test_segments.items():
    print(f"  {wave_type}: {segments}")

result = ecg.interpolate_intervals(test_segments)

print("\n" + "="*60)
print("Results after matching and outlier removal:")
print("="*60)

for interval_type in ['PR interval', 'QT interval', 'ST segment']:
    intervals = result[interval_type]
    print(f"\n{interval_type}:")
    print(f"  Count: {len(intervals)}")
    if intervals:
        lengths = [end - start for start, end in intervals]
        print(f"  Intervals: {intervals}")
        print(f"  Lengths: {lengths}")
        print(f"  Mean length: {np.mean(lengths):.2f}")
        print(f"  Median length: {np.median(lengths):.2f}")
        print(f"  Std dev: {np.std(lengths):.2f}")

print(f"\n\nMatched beats: {len(result['matched_beats'])}")
for i, (p, qrs, t) in enumerate(result['matched_beats']):
    print(f"  Beat {i+1}: P={p}, QRS={qrs}, T={t}")

print("\n" + "="*60)
print("Test completed successfully!")
