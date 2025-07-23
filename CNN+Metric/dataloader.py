#!/usr/bin/env python3
"""
Data loader for T-phase spectrogram classification

Handles loading and preprocessing of seismic spectrogram data for
deep learning models. Supports automatic train/test splitting and
class balancing.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import glob
import random

class TSpectrogramDataset(Dataset):
    """
    Dataset class for T-phase spectrogram data
    
    Loads .npy files containing 3-component spectrogram data and handles
    train/test splitting with stratification.
    """

    def __init__(self, data_root, class_names=None, transform=None, 
                 phase='train', test_size=0.2, random_state=42):
        """
        Initialize the dataset
        
        Args:
            data_root (str): Root directory containing class subdirectories
            class_names (list): List of class names (directory names)
            transform: Optional transform to apply to spectrograms
            phase (str): 'train' or 'test'
            test_size (float): Fraction of data to use for testing
            random_state (int): Random seed for reproducible splits
        """
        self.data_root = data_root
        self.transform = transform
        self.phase = phase
        self.random_state = random_state

        # Set default class names if not provided
        if class_names is None:
            self.class_names = ['Single T-Phase', 'Multiple T-Phase', 'No T-Phase']
        else:
            self.class_names = class_names

        # Create class name to index mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load all data paths and labels
        self._load_data()
        
        # Split data into train/test sets
        self._split_data(test_size, random_state)
        
        # Print dataset statistics
        self._print_statistics()

    def _load_data(self):
        """Load all data file paths and corresponding labels"""
        self.data_paths = []
        self.labels = []

        for class_name in self.class_names:
            class_dir = os.path.join(self.data_root, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory '{class_dir}' not found")
                continue

            # Find all .npy files in class directory
            npy_files = glob.glob(os.path.join(class_dir, "*.npy"))
            
            for npy_file in npy_files:
                self.data_paths.append(npy_file)
                self.labels.append(self.class_to_idx[class_name])

        # Convert to numpy arrays
        self.data_paths = np.array(self.data_paths)
        self.labels = np.array(self.labels, dtype=np.int64)

    def _split_data(self, test_size, random_state):
        """Split data into train and test sets"""
        if len(self.data_paths) == 0:
            print("Warning: No data found!")
            return
            
        # Perform stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            self.data_paths, self.labels, 
            test_size=test_size,
            random_state=random_state, 
            stratify=self.labels
        )

        # Select data based on phase
        if self.phase == 'train':
            self.data_paths = X_train
            self.labels = y_train
        else:  # test phase
            self.data_paths = X_test
            self.labels = y_test

    def _print_statistics(self):
        """Print dataset statistics"""
        print(f"\n{self.phase.capitalize()} Dataset Statistics:")
        print("-" * 40)
        
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            class_name = self.class_names[label]
            percentage = (count / len(self.labels)) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        print(f"  Total: {len(self.labels)} samples")

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data_paths)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: (spectrogram_tensor, label_tensor)
        """
        # Load spectrogram data
        spec_path = self.data_paths[idx]
        label = self.labels[idx]
        
        try:
            spec_data = np.load(spec_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load spectrogram from {spec_path}: {e}")

        # Convert to PyTorch tensor
        spec_data = torch.from_numpy(spec_data).float()
        
        # Handle different input formats
        if spec_data.dim() == 3:
            if spec_data.shape[-1] == 3:  # (F, T, C) format
                spec_data = spec_data.permute(2, 0, 1)  # Convert to (C, F, T)
            elif spec_data.shape[0] == 3:  # Already (C, F, T) format
                pass
            else:
                raise ValueError(f"Unexpected spectrogram shape: {spec_data.shape}")
        else:
            raise ValueError(f"Expected 3D spectrogram, got {spec_data.dim()}D")

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        # Apply transforms if provided
        if self.transform:
            spec_data = self.transform(spec_data)

        return spec_data, label

def seed_worker(worker_id):
    """Worker initialization function for reproducible data loading"""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloaders(data_root, batch_size=32, num_workers=4, 
                   class_names=None, test_size=0.2):
    """
    Create train and test data loaders
    
    Args:
        data_root (str): Root directory containing class subdirectories
        batch_size (int): Batch size for data loading
        num_workers (int): Number of worker processes for data loading
        class_names (list): List of class names
        test_size (float): Fraction of data to use for testing
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = TSpectrogramDataset(
        data_root=data_root,
        class_names=class_names, 
        phase='train',
        test_size=test_size
    )
    
    test_dataset = TSpectrogramDataset(
        data_root=data_root,
        class_names=class_names, 
        phase='test',
        test_size=test_size
    )
    
    # Create generator for reproducible data loading
    generator = torch.Generator()
    generator.manual_seed(42)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True,
        worker_init_fn=seed_worker, 
        generator=generator,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True,
        worker_init_fn=seed_worker, 
        generator=generator,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, test_loader
