# MetricTWaveNet

# Preprocessing: Waveform to Spectrogram

This module prepares seismic waveform data for classification by:

1. Extracting fixed-length segments around known event times
2. Applying bandpass filtering and amplitude normalization
3. Generating time-frequency spectrograms (via STFT)

---

## 🔧 Scripts

- `data_preprocessing.py`: Reads `.MSEED` + `.xlsx` and trims waveforms.
- `spectrogram_generation.py`: Generates 3-channel spectrograms and saves them as `.npy`.


