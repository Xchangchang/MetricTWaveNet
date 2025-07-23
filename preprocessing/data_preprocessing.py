#!/usr/bin/env python3
"""
Seismic Data Preprocessing Script

Before use, modify paths and parameters in config.py
"""
import os
import obspy
import numpy as np
from obspy import UTCDateTime
import pandas as pd
from config import *

def process_dataset():
    """Process dataset"""
    for class_name in CLASS_NAMES:
        print(f"Processing {class_name}...")
        
        excel_path = os.path.join(BASE_DIR, class_name, f"updated_{class_name}.xlsx")
        df = pd.read_excel(excel_path)
        
        output_dir = os.path.join(OUTPUT_BASE, class_name)
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, row in df.iterrows():
            try:
                mseed_file = os.path.join(BASE_DIR, class_name, f"{row['data']}.MSEED")
                st = process_single_file(mseed_file, row['event_time'])
                
                if st is not None:
                    output_file = os.path.join(output_dir, f"processed_{row['data']}.MSEED")
                    st.write(output_file, format='MSEED')
                    print(f"✓ {row['data']}")
                    
            except Exception as e:
                print(f"✗ {row['data']}: {e}")

def process_single_file(mseed_file, event_time):
    """Process single MSEED file"""
    try:
        st = obspy.read(mseed_file)
        start_time = UTCDateTime(event_time)
        end_time = start_time + TIME_WINDOW
        
        # Data preprocessing
        st.merge(fill_value=0)
        st.trim(start_time, end_time, pad=True, fill_value=0)
        
        # Trim to fixed length
        target_length = int(TIME_WINDOW * st[0].stats.sampling_rate)
        for tr in st:
            if len(tr.data) > target_length:
                tr.data = tr.data[:target_length]
        
        # Filtering and normalization
        for tr in st:
            tr.filter('bandpass', freqmin=FREQ_MIN, freqmax=FREQ_MAX, 
                     corners=FILTER_CORNERS, zerophase=True)
        
        max_amp = max([np.max(np.abs(tr.data)) for tr in st])
        if max_amp > 0:
            for tr in st:
                tr.data = tr.data / max_amp
                
        return st
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    process_dataset()


