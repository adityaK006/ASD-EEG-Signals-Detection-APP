# load_data.py
import os
import scipy.io
import numpy as np

def load_data(dataset_path):
    all_data = []
    all_labels = []
    
    # Loop through the dataset
    for subject in os.listdir(dataset_path):
        subject_folder = os.path.join(dataset_path, subject)
        if os.path.isdir(subject_folder):
            for session in os.listdir(subject_folder):
                session_folder = os.path.join(subject_folder, session)
                if os.path.isdir(session_folder):
                    # Load data for this subject-session
                    train_folder = os.path.join(session_folder, "Train")
                    train_data_path = os.path.join(train_folder, "trainData.mat")
                    data = scipy.io.loadmat(train_data_path)
                    
                    # Extract the EEG data
                    eeg_data = np.transpose(data['trainData'], (1, 2, 0))  # Shape: (epochs, timepoints, channels)
                    
                    # Load the labels
                    train_labels = np.loadtxt(os.path.join(train_folder, "trainLabels.txt"))
                    
                    # Store in lists
                    all_data.append(eeg_data)
                    all_labels.append(train_labels)
    
    # Convert lists to numpy arrays
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    print(f"Loaded Data Shape: {all_data.shape}")
    print(f"Loaded Labels Shape: {all_labels.shape}")
    
    return all_data, all_labels
