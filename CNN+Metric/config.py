#!/usr/bin/env python3
"""
Configuration file for T-phase seismic event classification

Contains all configuration parameters for data loading, model architecture,
and training hyperparameters.
"""

# Data configuration
DATA_CONFIG = {
    'data_root': "path/to/spectrogram/data",  # Path to spectrogram data
    'class_names':['No T-Phase','Single T-Phase', 'Multiple T-Phase']
}

# Model configuration
MODEL_CONFIG = {
    'embedding_dim': 128,  # Embedding vector dimension for metric learning
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 32,
    'num_workers': 4,
    'epochs': 100,
    'lr': 0.001,
    'weight_decay': 1e-5,
    'margin': 0.3,  # Triplet loss margin
    'triplet_weight': 1.0,  # Weight for triplet loss component
    'mining_strategy': 'batch-semi-hard',  # Options: 'batch-hard', 'batch-semi-hard', 'batch-all'
    'save_dir': './results',  # Base directory for saving results
    'viz_interval': 5,  # Visualization interval (epochs)
}

def get_config():
    """
    Merge all configuration dictionaries into a single config
    
    Returns:
        dict: Complete configuration dictionary
    """
    config = {}
    config.update(DATA_CONFIG)
    config.update(MODEL_CONFIG)
    config.update(TRAIN_CONFIG)
    return config

if __name__ == "__main__":
    config = get_config()
    print("Configuration settings:")
    print("-" * 30)
    for key, value in config.items():
        print(f"{key}: {value}")
