import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import wfdb
from models.ecgseg.utils import load_model, predict

def load_single_ecg(record_path, lead='ii'):
    """
    Load a specific lead from a single ECG record
    
    Args:
        record_path (str): ECG record file path (without extension)
        lead (str): Lead name, such as 'ii', 'v1', 'v2', etc.
    
    Returns:
        torch.Tensor: ECG data tensor with shape (1, 1, length)
    """
    # Read ECG record
    record = wfdb.rdrecord(record_path)
    
    # Get lead data
    lead_names = record.sig_name
    if lead not in lead_names:
        raise ValueError(f"Lead {lead} does not exist, available leads: {lead_names}")
    
    lead_idx = lead_names.index(lead)
    ecg_signal = record.p_signal[:, lead_idx]
    
    # Convert to tensor and add batch and channel dimensions
    ecg_tensor = torch.tensor(ecg_signal, dtype=torch.float32)
    ecg_tensor = ecg_tensor.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, length)
    
    return ecg_tensor

def segment_labels_to_names():
    """Return names corresponding to segmentation labels"""
    return {0: 'P wave', 1: 'QRS complex', 2: 'T wave', 3: 'Background'}

def extract_ecg_wave_segments(ecg_signal, model_path="models/ecgseg/ecgseg.pth"):
    """
    Extract ECG wave segments and their positions from a single lead ECG signal
    
    Args:
        ecg_signal (torch.Tensor or np.ndarray): ECG signal data, 1D array
        model_path (str): Path to the trained segmentation model
    
    Returns:
        dict: Dictionary with wave segment names as keys and list of (start, end) tuples as values
              Example: {'P wave': [(100, 150), (400, 450)], 'QRS complex': [(200, 280)], ...}
    """
    # Load model if not already loaded
    if not hasattr(extract_ecg_wave_segments, '_model'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        extract_ecg_wave_segments._model, _ = load_model(model_path, n_channels=32)
        print("Model loaded for segmentation")
    
    model = extract_ecg_wave_segments._model
    
    # Convert input to proper tensor format
    if isinstance(ecg_signal, np.ndarray):
        ecg_tensor = torch.tensor(ecg_signal, dtype=torch.float32)
    else:
        ecg_tensor = ecg_signal.clone().detach()
    
    # Ensure proper shape: (1, 1, length)
    if ecg_tensor.dim() == 1:
        ecg_tensor = ecg_tensor.unsqueeze(0).unsqueeze(0)
    elif ecg_tensor.dim() == 2:
        ecg_tensor = ecg_tensor.unsqueeze(0)
    
    # Perform segmentation prediction
    seg_output, _ = predict(model, ecg_tensor)
    
    # Convert predictions to segment labels
    seg_labels = torch.argmax(seg_output, dim=1).squeeze().cpu().numpy()
    segment_names = segment_labels_to_names()
    
    # Extract start-end positions for each wave type
    wave_segments = {}
    
    for label, name in segment_names.items():
        if name == 'Background':  # Skip background segments
            continue
            
        mask = seg_labels == label
        if mask.any():
            # Find start and end positions of the wave segments
            diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            # Create list of (start, end) tuples
            segments = [(int(start), int(end)) for start, end in zip(starts, ends)]
            wave_segments[name] = segments
        else:
            wave_segments[name] = []
    
    return wave_segments

def visualize_ecg_segmentation(ecg_data, seg_prediction, cls_prediction=None, 
                               title="ECG Segmentation Result", save_path=None):
    """
    Visualize ECG signal and segmentation results
    
    Args:
        ecg_data (torch.Tensor): ECG data with shape (1, 1, length)
        seg_prediction (torch.Tensor): Segmentation prediction with shape (1, 4, length)
        cls_prediction (torch.Tensor): Classification prediction with shape (1, 2)
        title (str): Chart title
        save_path (str): Save path, if None then display the image
    """
    # Convert to numpy arrays
    ecg_signal = ecg_data.squeeze().cpu().numpy()
    seg_probs = torch.softmax(seg_prediction, dim=1).squeeze().cpu().numpy()
    seg_labels = torch.argmax(seg_prediction, dim=1).squeeze().cpu().numpy()
    
    # Classification results
    if cls_prediction is not None:
        cls_probs = torch.softmax(cls_prediction, dim=1).cpu().numpy()
        cls_pred = torch.argmax(cls_prediction, dim=1).item()
        cls_names = {0: 'Normal Rhythm', 1: 'Atrial Fibrillation/Flutter'}
    
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot original ECG signal
    axes[0].plot(ecg_signal, 'b-', linewidth=1)
    axes[0].set_title(f'{title} - Original ECG Signal', fontsize=14)
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Plot segmentation probabilities
    segment_names = segment_labels_to_names()
    colors = ['red', 'green', 'orange', 'gray']
    
    for i, (label, name) in enumerate(segment_names.items()):
        axes[1].plot(seg_probs[i], color=colors[i], label=name, linewidth=2)
    
    axes[1].set_title('Wave Segment Probability Distribution', fontsize=14)
    axes[1].set_ylabel('Probability', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Plot final segmentation result
    axes[2].plot(ecg_signal, 'b-', linewidth=1, alpha=0.7, label='ECG Signal')
    
    # Add background colors for different wave segments
    for i, (label, name) in enumerate(segment_names.items()):
        mask = seg_labels == i
        if mask.any():
            axes[2].fill_between(range(len(ecg_signal)), 
                               ecg_signal.min(), ecg_signal.max(), 
                               where=mask, alpha=0.3, color=colors[i], label=name)
    
    axes[2].set_title('ECG Wave Segment Segmentation Result', fontsize=14)
    axes[2].set_xlabel('Time Points', fontsize=12)
    axes[2].set_ylabel('Amplitude', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # Add classification result text
    if cls_prediction is not None:
        fig.suptitle(f'{title}\nClassification Result: {cls_names[cls_pred]} (Confidence: {cls_probs[0][cls_pred]:.3f})', 
                    fontsize=16, y=0.95)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Result saved to: {save_path}")
    else:
        plt.show()

def analyze_segmentation_results(seg_prediction, sampling_rate=500):
    """
    Analyze segmentation results, calculate duration and position of each wave segment
    
    Args:
        seg_prediction (torch.Tensor): Segmentation prediction result
        sampling_rate (int): Sampling rate
    
    Returns:
        dict: Analysis results
    """
    seg_labels = torch.argmax(seg_prediction, dim=1).squeeze().cpu().numpy()
    segment_names = segment_labels_to_names()
    
    results = {}
    for label, name in segment_names.items():
        mask = seg_labels == label
        if mask.any():
            # Find start and end positions of the wave segment
            diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            segments = []
            for start, end in zip(starts, ends):
                duration_ms = (end - start) / sampling_rate * 1000
                segments.append({
                    'start_sample': start,
                    'end_sample': end,
                    'duration_ms': duration_ms
                })
            
            results[name] = {
                'count': len(segments),
                'segments': segments,
                'total_duration_ms': sum(s['duration_ms'] for s in segments)
            }
    
    return results

def main():
    """Main function: Demonstrate the usage of ECG segmentation model"""
    
    # 1. Load trained model
    print("Loading trained model...")
    model_path = "./checkpoints/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist")
        print("Please ensure the model has been trained or use the correct model path")
        return
    
    model, checkpoint_info = load_model(model_path, n_channels=32)
    print(f"Model loaded successfully! Training epoch: {checkpoint_info['epoch']}")
    
    # 2. Load test ECG data
    print("\nLoading ECG data...")
    data_dir = "./data/lobachevsky-university-electrocardiography-database-1.0.1/data"
    
    # Use the first available ECG record as example
    available_files = [f for f in os.listdir(data_dir) if f.endswith('.hea')]
    if not available_files:
        print("Error: No ECG data files found")
        return
    
    # Select the first file as example
    test_file = os.path.join(data_dir, available_files[0])[:-4]  # Remove .hea extension
    print(f"Using test file: {available_files[0]}")
    
    # Load ECG data (using lead II)
    try:
        ecg_data = load_single_ecg(test_file, lead='ii')
        print(f"ECG data shape: {ecg_data.shape}")
    except Exception as e:
        print(f"Error loading ECG data: {e}")
        return
    
    # 3. Perform prediction
    print("\nPerforming ECG segmentation prediction...")
    seg_output, cls_output = predict(model, ecg_data)
    
    print(f"Segmentation output shape: {seg_output.shape}")  # (1, 4, length)
    print(f"Classification output shape: {cls_output.shape}")  # (1, 2)
    
    # 4. Analyze results
    print("\nAnalyzing segmentation results...")
    analysis_results = analyze_segmentation_results(seg_output)
    
    for segment_name, info in analysis_results.items():
        print(f"\n{segment_name}:")
        print(f"  Number of detected segments: {info['count']}")
        print(f"  Total duration: {info['total_duration_ms']:.1f} milliseconds")
        
        for i, segment in enumerate(info['segments'][:3]):  # Only show first 3
            print(f"  Segment{i+1}: samples {segment['start_sample']}-{segment['end_sample']} "
                  f"(duration {segment['duration_ms']:.1f}ms)")
        
        if len(info['segments']) > 3:
            print(f"  ... and {len(info['segments'])-3} more segments")
    
    # 5. Visualize results
    print("\nGenerating visualization results...")
    visualize_ecg_segmentation(
        ecg_data, 
        seg_output, 
        cls_output,
        title=f"ECG Segmentation Example - {available_files[0]}",
        save_path="./ecg_segmentation_result.png"
    )
    
    # 6. Display classification results
    cls_probs = torch.softmax(cls_output, dim=1).cpu().numpy()
    cls_pred = torch.argmax(cls_output, dim=1).item()
    cls_names = {0: 'Normal Rhythm', 1: 'Atrial Fibrillation/Flutter'}
    
    print(f"\nClassification results:")
    print(f"  Predicted class: {cls_names[cls_pred]}")
    print(f"  Confidence: {cls_probs[0][cls_pred]:.3f}")
    print(f"  Class probabilities: Normal Rhythm={cls_probs[0][0]:.3f}, Atrial Fibrillation/Flutter={cls_probs[0][1]:.3f}")
    
    print("\nAnalysis complete! Result image saved as 'ecg_segmentation_result.png'")
    
    # 7. Test the new extract_ecg_wave_segments function
    print("\n" + "="*50)
    print("Testing extract_ecg_wave_segments function:")
    print("="*50)
    
    # Extract just the 1D signal for the new function
    ecg_1d = ecg_data.squeeze().cpu().numpy()
    wave_segments = extract_ecg_wave_segments(ecg_1d)
    
    print("\nExtracted wave segments:")
    for wave_type, segments in wave_segments.items():
        print(f"\n{wave_type}:")
        if segments:
            for i, (start, end) in enumerate(segments):
                duration = end - start
                print(f"  Segment {i+1}: ({start}, {end}) - duration: {duration} samples")
        else:
            print("  No segments found")
    
    # Example: How to use specific segments
    if 'QRS complex' in wave_segments and wave_segments['QRS complex']:
        first_qrs = wave_segments['QRS complex'][0]
        start, end = first_qrs
        print(f"\nFirst QRS complex is located at samples {start}-{end}")
        print(f"QRS signal values: {ecg_1d[start:end][:5]}...")  # Show first 5 values

if __name__ == "__main__":
    main()