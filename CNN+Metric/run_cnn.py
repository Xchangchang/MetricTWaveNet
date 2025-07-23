#!/usr/bin/env python3
"""
Run CNN Training Experiments

Runs CNN baseline experiments with multiple random seeds for robust results.
"""
import os
import torch
import random
import numpy as np
from datetime import datetime
import json
from config import get_config
from trainer_cnn import CNNTrainer

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def run_experiment(seed, config):
    """Run single experiment with given seed"""
    set_seed(seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['save_dir'] = os.path.join('./results', f"run_cnn_seed_{seed}_{timestamp}")
    
    # Save config
    os.makedirs(config['save_dir'], exist_ok=True)
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    config['seed'] = seed
    trainer = CNNTrainer(config)
    print(f"Starting CNN training (seed={seed}), results will be saved to: {config['save_dir']}")
    best_acc = trainer.train()
    print(f"CNN training completed (seed={seed}), best accuracy: {best_acc:.4f}")
    return best_acc

def main():
    """Main function"""
    SEEDS = [42, 123, 456]
    config = get_config()
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    accuracies = []
    for seed in SEEDS:
        print(f"\n=== Running CNN experiment (seed={seed}) ===")
        best_acc = run_experiment(seed, config.copy())
        accuracies.append(best_acc)

    # Summary
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"\n{'='*50}")
    print("CNN TRAINING SUMMARY")
    print(f"{'='*50}")
    print(f"Individual accuracies: {[f'{acc:.4f}' for acc in accuracies]}")
    print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

if __name__ == "__main__":
    main()
