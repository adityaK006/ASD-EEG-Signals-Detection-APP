import os
import numpy as np
from sklearn.model_selection import train_test_split
from load_data import load_data
# from visualize_data import visualize_eeg_signal
from load_normal_data import load_normal_data
from preprocess_normal_data import preprocess_normal_data
# from visualize_normalpreprocessed_eeg import visualize_preprocessed_eeg

def extract_features(X):
    """
    Convert EEG data (batch, epochs, 1600, channels) into (batch, epochs, features)
    by computing mean & variance per channel.
    """
    mean_features = np.mean(X, axis=2)  # Shape: (batch, epochs, channels)
    var_features = np.var(X, axis=2)  # Shape: (batch, epochs, channels)

    # Concatenate features along the last axis
    return np.concatenate([mean_features, var_features], axis=-1)  # Shape: (batch, epochs, 2 * channels)

if __name__ == "__main__":
    dataset_path = r"E:\COLLEGE\DL Project\data" 
    normal_dataset_path = r"E:\COLLEGE\DL Project\SPIS-Resting-State-Dataset-master\Pre-Sart EEG"
    
    # Step 1: Load Autism EEG data
    autism_eeg_data, autism_labels = load_data(dataset_path)  # Shape: (105, 350, 1600, 8)

    # Step 2: Load and preprocess Normal EEG data
    normal_eeg_data, _ = load_normal_data(normal_dataset_path)
    preprocessed_normal_eeg, normal_labels = preprocess_normal_data(normal_eeg_data)  # Shape: (20, 350, 1600, 8)

    # Step 3: Merge Autism and Normal Data
    X = np.concatenate((autism_eeg_data, preprocessed_normal_eeg), axis=0)  # Shape: (125, 350, 1600, 8)
    y = np.concatenate((np.ones(autism_eeg_data.shape[0]), np.zeros(preprocessed_normal_eeg.shape[0])), axis=0)  # Labels: 1 for Autism, 0 for Normal

    print(f"✅ Final Merged Data Shape: {X.shape}")  # Expected: (125, 350, 1600, 8)
    print(f"✅ Final Labels Shape: {y.shape}")  # Expected: (125,)

    # Step 4: Train-Validation-Test Split (70%-15%-15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    print(f"✅ Training Data Shape: {X_train.shape}")  # ~70% of data
    print(f"✅ Validation Data Shape: {X_val.shape}")  # ~15% of data
    print(f"✅ Testing Data Shape: {X_test.shape}")  # ~15% of data

    # Step 5: Feature Extraction
    X_train_features = extract_features(X_train)  # Shape: (87, 350, 16)
    X_val_features = extract_features(X_val)  # Shape: (19, 350, 16)
    X_test_features = extract_features(X_test)  # Shape: (19, 350, 16)

    print(f"✅ Processed Training Data Shape: {X_train_features.shape}")  # Expected: (87, 350, 16)
    print(f"✅ Processed Validation Data Shape: {X_val_features.shape}")  # Expected: (19, 350, 16)
    print(f"✅ Processed Testing Data Shape: {X_test_features.shape}")  # Expected: (19, 350, 16)

# Add this at the very bottom of the script
def get_raw_data():
    return autism_eeg_data, preprocessed_normal_eeg
    # # Save data for model training
    # np.save("train_X.npy", X_train_features)
    # np.save("train_y.npy", y_train)
    # np.save("val_X.npy", X_val_features)
    # np.save("val_y.npy", y_val)
    # np.save("test_X.npy", X_test_features)
    # np.save("test_y.npy", y_test)
