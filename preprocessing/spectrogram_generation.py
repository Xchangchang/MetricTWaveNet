#!/usr/bin/env python3
"""
Spectrogram Generation Script

Before use, modify paths and parameters in config.py
"""
import os
import obspy
import numpy as np
from scipy import signal
from config import *

def generate_spectrograms():
    """Generate spectrograms"""
    for class_name in CLASS_NAMES:
        print(f"Processing {class_name} spectrograms...")
        
        output_dir = os.path.join(SPECTROGRAM_BASE, class_name)
        os.makedirs(output_dir, exist_ok=True)
        
        mseed_dir = os.path.join(OUTPUT_BASE, class_name)
        
        for file in os.listdir(mseed_dir):
            if file.endswith('.MSEED'):
                try:
                    st = obspy.read(os.path.join(mseed_dir, file))
                    
                    if len(st) != 3:
                        print(f"Warning: {file} - not 3 components")
                        continue
                        
                    specs = create_spectrogram(st)
                    if specs is not None:
                        output_file = os.path.join(output_dir, file.replace('.MSEED', '.npy'))
                        np.save(output_file, specs)
                        print(f"✓ {file}")
                        
                except Exception as e:
                    print(f"✗ {file}: {e}")

def create_spectrogram(st):
    """Create 3-component spectrogram"""
    specs = []
    
    for tr in st:
        try:
            f, t, Sxx = signal.stft(
                tr.data,
                fs=tr.stats.sampling_rate,
                nperseg=int(tr.stats.sampling_rate * STFT_WINDOW),
                noverlap=int(tr.stats.sampling_rate * STFT_OVERLAP),
                window='hann',
                boundary='zeros'
            )
            
            freq_mask = (f >= FREQ_MIN) & (f <= FREQ_MAX)
            spec = np.abs(Sxx[freq_mask])
            
            if np.max(spec) > 0:
                spec = spec / np.max(spec)
                
            specs.append(spec)
            
        except Exception as e:
            print(f"STFT error: {e}")
            return None
            
    return np.stack(specs) if len(specs) == 3 else None

if __name__ == "__main__":
    generate_spectrograms()


