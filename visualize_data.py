# visualize_data.py
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_data

def visualize_eeg_signal(data, subject_index=0, session_index=0, channel_index=0):
    # Select the EEG signal to visualize
    eeg_signal = data[subject_index, session_index, :, channel_index]
    timepoints = np.linspace(-200, 1200, eeg_signal.shape[0])  # Time from -200 ms to 1200 ms
    
    # Plot the EEG signal
    plt.plot(timepoints, eeg_signal)
    plt.title(f"EEG Signal - Subject {subject_index+1}, Session {session_index+1}, Channel {channel_index+1}")
    plt.xlabel("Time (ms)")
    plt.ylabel("EEG Amplitude")
    plt.show()

if __name__ == "__main__":
    dataset_path = "./data"  # Update this with the path to your 'data' folder
    eeg_data, _ = load_data(dataset_path)  # Load the data
    
    # Example: Visualize EEG for subject 1, session 1, and channel 1
    visualize_eeg_signal(eeg_data, subject_index=0, session_index=0, channel_index=0)
