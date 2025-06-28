import numpy as np
import torch
from model import EEG_RNN  # replace with your actual model class filename

# Load saved model
model = EEG_RNN()
model.load_state_dict(torch.load("eeg_rnn_best_model18april.pth"))
model.eval()

# ✅ Step 1: Load preprocessed training data
X_train = np.load("train_X_balanced.npy")  # Shape: (146, 350, 16)
y_train = np.load("train_y_balanced.npy")  # Shape: (146,)

# ✅ Step 2: Select a real autism EEG sample
autism_indices = np.where(y_train == 1)[0]  # Indices of autism samples
real_autism_sample = X_train[autism_indices[0]]  # Shape: (350, 16)

# ✅ Step 3: Add small Gaussian noise to simulate unseen-like variation
np.random.seed(42)
noise = np.random.normal(0, 0.05, real_autism_sample.shape)  # Mean 0, SD 0.05
synthetic_sample = real_autism_sample + noise  # Shape: (350, 16)

# ✅ Step 4: Convert to tensor and reshape for RNN
synthetic_tensor = torch.tensor(synthetic_sample, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 350, 16)

# ✅ Step 5: Predict using the loaded model
with torch.no_grad():
    output = model(synthetic_tensor)
    prediction = torch.argmax(output, dim=1).item()

# ✅ Step 6: Output the result
print("🧠 Predicted class for realistic synthetic autism EEG:", prediction)
