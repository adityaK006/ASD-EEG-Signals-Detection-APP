import numpy as np
import scipy.signal

# Define the required channels
AUTISM_CHANNELS = ['C3', 'Cz', 'C4', 'CPz', 'P3', 'Pz', 'P4', 'POz']

# Mapping from the normal dataset's 64 channels to their indices
NORMAL_CHANNELS_LIST = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5',
                        'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz',
                        'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'Afz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8',
                        'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2',
                        'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

# Get the indices of the required channels
SELECTED_CHANNELS_IDX = [NORMAL_CHANNELS_LIST.index(ch) for ch in AUTISM_CHANNELS]

def preprocess_normal_data(eeg_data, sampling_rate=256, target_rate=250, timepoints_per_epoch=1600):
    """
    Preprocess normal EEG data:
    - Resample to 250 Hz using polyphase method (better for EEG)
    - Extract required 8 channels
    - Segment into 350 epochs with proper overlap
    """
    eeg_data = np.squeeze(eeg_data)  
    eeg_data = eeg_data[:, :64, :]  # Keep only EEG channels
    eeg_data = np.swapaxes(eeg_data, 1, 2)  # Shape: (samples, timepoints, channels)

    # Resample using polyphase filter (more stable)
    eeg_data = scipy.signal.resample_poly(eeg_data, up=target_rate, down=sampling_rate, axis=1)

    # Extract selected 8 channels
    eeg_data = eeg_data[:, :, SELECTED_CHANNELS_IDX]  # Shape: (samples, timepoints, 8)

    # Segmenting with overlap to get 350 epochs
    total_timepoints = eeg_data.shape[1]
    desired_epochs = 350
    stride_float = (total_timepoints - timepoints_per_epoch) / (desired_epochs - 1)

    segmented_eeg = np.zeros((eeg_data.shape[0], desired_epochs, timepoints_per_epoch, len(AUTISM_CHANNELS)))

    for ep in range(desired_epochs):
        start = int(round(ep * stride_float))
        end = start + timepoints_per_epoch
        segmented_eeg[:, ep, :, :] = eeg_data[:, start:end, :]

    # Labels (0 = normal)
    all_labels = np.zeros((segmented_eeg.shape[0], desired_epochs))

    print(f"✅ Final Preprocessed EEG Shape: {segmented_eeg.shape}")  # (20, 350, 1600, 8)
    print(f"✅ Updated Labels Shape: {all_labels.shape}")  # (20, 350)

    return segmented_eeg, all_labels
