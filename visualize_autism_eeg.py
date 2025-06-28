import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data  # ✅ Make sure this file exists

# ✅ Step 1: Load preprocessed autism EEG data
autism_data, _ = load_data(r"E:\COLLEGE\DL Project\data")  # Your loader should return shape (105, 350, 1600, 8)

# ✅ Step 2: Select a subject and epoch for visualization
subject_idx = 0  # You can change to any value < 105
epoch_idx = 0    # You can change to any value < 350
sample = autism_data[subject_idx, epoch_idx]  # Shape: (1600, 8)

# ✅ Step 3: Plot all 8 channels
plt.figure(figsize=(14, 10))
for ch in range(8):
    plt.subplot(8, 1, ch + 1)
    plt.plot(sample[:, ch], color='royalblue')
    plt.title(f"Channel {ch + 1}", fontsize=9)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()

plt.suptitle("✅ Preprocessed Autism EEG Signal (Subject 1, Epoch 1)", fontsize=14, y=1.02)
plt.show()

# ✅ Step 4: Plot one channel in clean style (e.g., Channel 1)
channel_idx = 0
def plot_eeg(signal, title="EEG Signal", sample_rate=256):
    plt.figure(figsize=(10, 4))
    time_axis = np.arange(len(signal)) / sample_rate
    plt.plot(time_axis, signal, label="EEG Signal", color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.show()

plot_eeg(sample[:, channel_idx], title="Autism EEG Signal")