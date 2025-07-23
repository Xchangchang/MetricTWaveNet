#!/usr/bin/env python3
"""
Configuration Template File for MetricTWaveNet

Instructions:
1. Copy this file to `config.py` in the same directory:
       cp config_template.py config.py
2. Modify the paths below to match your local file system.
3. This file is required by:
   - preprocessing/data_preprocessing.py
   - preprocessing/spectrogram_generation.py
"""

# ======== PATH SETTINGS ========
# Directory where raw waveform files and metadata (.MSEED, .xlsx) are stored
BASE_DIR = "./data/raw"

# Where to save processed waveform segments
OUTPUT_BASE = "./data/processed"

# Where to save spectrogram .npy files
SPECTROGRAM_BASE = "./data/spectrogram"

# ======== DATASET LABELS ========
# Categories used for labeling the samples
CLASS_NAMES = ['Single T-Phase', 'Multiple T-Phase', 'No T-Phase']


# ======== WAVEFORM PROCESSING PARAMETERS ========
# Total time window (in seconds) for each sample
TIME_WINDOW = 1500

# Bandpass filter range
FREQ_MIN = 2
FREQ_MAX = 4
FILTER_CORNERS = 4

# ======== SPECTROGRAM SETTINGS ========
# Short-Time Fourier Transform window and overlap
STFT_WINDOW = 10
STFT_OVERLAP = 8
