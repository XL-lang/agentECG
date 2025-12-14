import torch
from .model import ECGUNet3pCGM

def load_model(checkpoint_path, n_channels=32, device=None):
    """
    Load a trained ECG model from checkpoint
    
    Args:
        checkpoint_path (str): Path to the model checkpoint file
        n_channels (int): Number of channels in first encoder feature map
        device (str or torch.device): Device to load the model on
        
    Returns:
        model: Loaded ECG model
        checkpoint_info (dict): Information about the checkpoint (epoch, losses, etc.)
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = ECGUNet3pCGM(n_channels=n_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Extract checkpoint information
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'train_loss': checkpoint.get('train_loss', 'unknown'),
        'test_loss': checkpoint.get('test_loss', 'unknown'),
        'test_loss_seg': checkpoint.get('test_loss_seg', 'unknown'),
        'test_loss_cls': checkpoint.get('test_loss_cls', 'unknown'),
    }
    
    print(f"Model loaded from epoch {checkpoint_info['epoch']}")
    print(f"Test loss: {checkpoint_info['test_loss']}")
    
    return model, checkpoint_info

def predict(model, ecg_data, device=None):
    """
    Make predictions using the loaded model
    
    Args:
        model: Loaded ECG model
        ecg_data (torch.Tensor): ECG data tensor of shape (batch_size, 1, sequence_length)
        device (str or torch.device): Device to run inference on
        
    Returns:
        seg_output: Segmentation predictions
        cls_output: Classification predictions
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    ecg_data = ecg_data.to(device)
    
    with torch.no_grad():
        seg_output, cls_output = model(ecg_data)
    
    return seg_output, cls_output