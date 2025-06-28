# import numpy as np
# import matplotlib.pyplot as plt

# def visualize_preprocessed_eeg(eeg_data, subject_idx=0, num_epochs=5):
#     """
#     Visualizes EEG signals from the preprocessed normal data.
    
#     Parameters:
#     - eeg_data: numpy array of shape (subjects, epochs, timepoints, channels)
#     - subject_idx: The subject whose EEG data to visualize (default: 0)
#     - num_epochs: Number of random epochs to plot (default: 5)
#     """
    
#     # Ensure subject index is within range
#     if subject_idx >= eeg_data.shape[0]:
#         raise ValueError(f"Subject index out of range. Max index: {eeg_data.shape[0] - 1}")

#     # Select the subject's data
#     subject_eeg = eeg_data[subject_idx]  # Shape: (350, 1600, 8)

#     # Randomly select epochs to visualize
#     epoch_indices = np.random.choice(subject_eeg.shape[0], num_epochs, replace=False)
    
#     # Define channel names
#     channels = ['C3', 'Cz', 'C4', 'CPz', 'P3', 'Pz', 'P4', 'POz']
    
#     fig, axes = plt.subplots(num_epochs, 1, figsize=(12, 8), sharex=True)

#     for i, epoch_idx in enumerate(epoch_indices):
#         epoch_data = subject_eeg[epoch_idx]  # Shape: (1600, 8)
#         time_axis = np.linspace(0, 1600, num=1600)  # Timepoints (1.6s window)

#         for ch in range(epoch_data.shape[1]):
#             axes[i].plot(time_axis, epoch_data[:, ch], label=channels[ch], alpha=0.7)

#         axes[i].set_title(f"Subject {subject_idx} - Epoch {epoch_idx}")
#         axes[i].legend(loc="upper right", fontsize=7)
    
#     plt.xlabel("Timepoints")
#     plt.suptitle(f"EEG Visualization for Subject {subject_idx}")
#     plt.show()


#visualizing 8 channels of normal preprocessed data
import numpy as np
import matplotlib.pyplot as plt
from preprocess_normal_data import preprocess_normal_data
from load_normal_data import load_normal_data

# ✅ Path to your SPIS normal EEG dataset
normal_dataset_path = r"E:\COLLEGE\DL Project\SPIS-Resting-State-Dataset-master\Pre-Sart EEG"

# ✅ Step 1: Load raw normal EEG data
normal_raw, _ = load_normal_data(normal_dataset_path)

# ✅ Step 2: Preprocess the normal EEG data
# Output shape: (subjects, epochs, 1600 time points, 8 channels)
preprocessed_normal_eeg, _ = preprocess_normal_data(normal_raw)

# ✅ Step 3: Select a sample for visualization
subject_idx = 0  # You can change to visualize a different subject
epoch_idx = 0
sample = preprocessed_normal_eeg[subject_idx, epoch_idx]  # Shape: (1600, 8)

# ✅ Step 4: Plot EEG signal from all 8 channels
plt.figure(figsize=(14, 10))
for ch in range(8):
    plt.subplot(8, 1, ch + 1)
    plt.plot(sample[:, ch], color='royalblue')
    plt.title(f"Channel {ch + 1}", fontsize=9)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()

plt.suptitle("✅ Preprocessed Normal EEG Signal (Subject 1, Epoch 1)", fontsize=14, y=1.02)
plt.show()

# ✅ Step 5: Plot in same style as Autism EEG plot
# Use the first channel (you can change this to any channel 0–7)
channel_idx = 0
time_axis = np.linspace(0, 1600 / 1145, 1600)  # Assuming 1145 Hz sampling rate

plt.figure(figsize=(12, 4))
plt.plot(time_axis, sample[:, channel_idx], label="EEG Signal", color="blue")
plt.title("Normal EEG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

