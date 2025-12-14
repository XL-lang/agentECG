#!/usr/bin/env python3
"""
Example usage of EcgQaDataset - PyTorch IterableDataset for ECG-QA data.
"""

import torch
from torch.utils.data import DataLoader
import sys
import os

# Add the data directory to Python path
sys.path.append('/home/xl/agentECG/data')

from EcgQaData import EcgQaDataset, create_ecg_qa_dataloader


def example_basic_usage():
    """Basic example of using EcgQaDataset."""
    print("=== Basic Usage Example ===")
    
    # Initialize dataset (without loading ECG signals for faster testing)
    dataset = EcgQaDataset(
        data_dir="/home/xl/agentECG/dataset/ecgqa_ptbxl",
        split="train",
        data_type="template",
        load_ecg_data=False  # Set to True to load actual ECG signals
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Iterate through first 3 samples
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        
        print(f"\nSample {i+1}:")
        print(f"  Question: {sample['question'][:100]}...")
        print(f"  Answer: {sample['answer']}")
        print(f"  Question Type: {sample['question_type']}")
        print(f"  ECG IDs: {sample['ecg_id']}")


def example_with_ecg_data():
    """Example loading actual ECG signals."""
    print("\n=== ECG Data Loading Example ===")
    
    # Initialize dataset with ECG data loading
    dataset = EcgQaDataset(
        data_dir="/home/xl/agentECG/dataset/ecgqa_ptbxl",
        split="train",
        data_type="template",
        ptbxl_data_dir="/home/xl/agentECG/dataset/dataset_1/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
        load_ecg_data=True,
        max_length=2500  # Shorter for demonstration
    )
    
    # Get one sample with ECG data
    for i, sample in enumerate(dataset):
        if 'ecg_data' in sample:
            print(f"ECG Data Shape: {sample['ecg_data'].shape}")
            print(f"Question: {sample['question'][:80]}...")
            break
        if i >= 10:  # Don't search too long
            print("No ECG data found in first 10 samples")
            break


def example_dataloader():
    """Example using DataLoader with custom collate function."""
    print("\n=== DataLoader Example ===")
    
    # Create DataLoader using helper function
    dataloader = create_ecg_qa_dataloader(
        data_dir="/home/xl/agentECG/dataset/ecgqa_ptbxl",
        split="train",
        data_type="template",
        batch_size=4,
        load_ecg_data=False,  # Faster for demonstration
        num_workers=0  # Use 0 for IterableDataset
    )
    
    # Get one batch
    for batch in dataloader:
        print(f"Batch size: {len(batch['questions'])}")
        print(f"First question: {batch['questions'][0][:60]}...")
        print(f"First answer: {batch['answers'][0]}")
        
        if 'ecg_data' in batch:
            print(f"ECG batch shape: {batch['ecg_data'].shape}")
        break


def example_different_splits():
    """Example showing different data splits and types."""
    print("\n=== Different Splits and Types Example ===")
    
    splits = ["train", "valid", "test"]
    data_types = ["template", "paraphrased"]
    
    for data_type in data_types:
        for split in splits:
            try:
                dataset = EcgQaDataset(
                    data_dir="/home/xl/agentECG/dataset/ecgqa_ptbxl",
                    split=split,
                    data_type=data_type,
                    load_ecg_data=False
                )
                print(f"{data_type.capitalize()} {split}: {len(dataset)} samples")
            except ValueError as e:
                print(f"{data_type.capitalize()} {split}: Not available ({str(e)})")


if __name__ == "__main__":
    print("ECG-QA IterableDataset Examples")
    print("=" * 40)
    
    try:
        example_basic_usage()
        example_different_splits()
        example_dataloader()
        
        # Uncomment to test ECG data loading (requires proper paths)
        # example_with_ecg_data()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure the data paths are correct and the dataset exists.")