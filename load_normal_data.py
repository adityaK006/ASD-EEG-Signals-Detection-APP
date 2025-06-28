import os
import scipy.io
import numpy as np

def load_normal_data(dataset_path):
    all_data = []
    all_labels = []
    
    for file in os.listdir(dataset_path):
        if file.endswith(".mat") and "EOG" not in file:
            file_path = os.path.join(dataset_path, file)
            data = scipy.io.loadmat(file_path)
            
            key = list(data.keys())[-1]
            eeg_data = np.array(data[key])

            if eeg_data.ndim == 2:
                eeg_data = eeg_data[np.newaxis, :, :]
            
            all_data.append(eeg_data)
            num_epochs = eeg_data.shape[0]

            # Instead of extend(), use np.zeros directly for consistent shape
            all_labels.append(np.zeros((1, num_epochs))) 

    all_data = np.array(all_data)  
    all_labels = np.vstack(all_labels)  # Stack labels properly to (subjects, epochs)

    print(f"Loaded Normal Data Shape: {all_data.shape}")  # Expected: (20, 107, 350, 8)
    print(f"Loaded Normal Labels Shape: {all_labels.shape}")  # Expected: (20, 107)

    return all_data, all_labels
