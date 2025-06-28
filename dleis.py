#install mne, pyqt6



import numpy as np
import scipy.io
import mne
import matplotlib
matplotlib.use('QtAgg')  # Ensure correct backend

# Load EEG data
data = scipy.io.loadmat(r"E:\COLLEGE\DL\data\SBJ01\S01\Train\trainData.mat")  
eeg_data = data['trainData']  # Shape: [channels x epochs x timepoints]

print("EEG Data Shape:", eeg_data.shape)

# Define channel names
channel_names = ['C3', 'Cz', 'C4', 'CPz', 'P3', 'Pz', 'P4', 'POz']
sfreq = 250  # Sampling frequency (adjust if needed)

# Create MNE Info object
info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")

# Reshape data to fit MNE's format: (n_channels, n_times)
epoch = 0  # Select one epoch
raw_data = eeg_data[:, epoch, :]  # Shape: (8, 1600)

# Convert to MNE RawArray
raw = mne.io.RawArray(raw_data, info)

# Plot the EEG signals (force interactive mode)
raw.plot(title="EEG Signals from One Epoch", scalings="auto", block=True)

