#!/usr/bin/env python3
"""
Scattering Transform Feature Extraction Script

Extracts scattering transform features from preprocessed MSEED files
for seismic T-phase classification.
"""
import os
import numpy as np
import obspy
from tqdm import tqdm
import kymatio.numpy as knp
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_BASE = "path/to/processed/data"  # Path to preprocessed MSEED files
CLASSES = ['Single T-Phase', 'Multiple T-Phase', 'No T-Phase']
OUTPUT_DIR = "path/to/scattering/features"  # Output directory for features
BEST_J = 8  # Scattering transform parameter
BEST_Q = 12  # Scattering transform parameter

def load_mseed_data(file_path):
    """Load MSEED file and return 3-component data"""
    try:
        st = obspy.read(file_path)
        if len(st) != 3:
            return None
        components = [tr.data for tr in st]
        return np.array(components)
    except:
        return None

def extract_scattering_features(signal_data, J=BEST_J, Q=BEST_Q):
    """Extract scattering transform features from 3-component seismic data"""
    signal_length = signal_data.shape[1]
    scattering = knp.Scattering1D(J, signal_length, Q)

    all_features = []
    for component in signal_data:
        Sx = scattering(component)
        Sx_avg = np.mean(Sx, axis=-1).squeeze()
        all_features.append(Sx_avg)

    combined_features = np.concatenate(all_features)
    return combined_features

def main():
    """Main feature extraction function"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_files = []
    all_labels = []

    # Collect all files and labels
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATA_BASE, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found")
            continue
            
        mseed_files = [f for f in os.listdir(class_dir) if f.endswith('.MSEED')]
        file_paths = [os.path.join(class_dir, f) for f in mseed_files]

        all_files.extend(file_paths)
        all_labels.extend([class_idx] * len(file_paths))

    all_files = np.array(all_files)
    all_labels = np.array(all_labels)

    # Extract features
    all_features = []
    valid_labels = []

    for file_path, label in tqdm(zip(all_files, all_labels), 
                                desc="Extracting features", total=len(all_files)):
        components = load_mseed_data(file_path)
        if components is None:
            continue
        features = extract_scattering_features(components)
        all_features.append(features)
        valid_labels.append(label)

    all_features = np.array(all_features)
    valid_labels = np.array(valid_labels)

    # Save features
    np.save(os.path.join(OUTPUT_DIR, 'scatter_features.npy'), all_features)
    np.save(os.path.join(OUTPUT_DIR, 'labels.npy'), valid_labels)

    print(f"Feature extraction completed: {all_features.shape[0]} samples, "
          f"feature dim = {all_features.shape[1]}")
    print(f"Features saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
