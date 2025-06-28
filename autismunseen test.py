import torch
import numpy as np
from model import EEG_RNN  # Replace with your model file name
import matplotlib.pyplot as plt

# Step 1: Load your trained model
model = EEG_RNN()
model.load_state_dict(torch.load("eeg_rnn_best_model18april.pth"))
model.eval()

# Step 2: Generate synthetic EEG signal resembling autism patterns
# We'll use mean and std from autism samples (you can tune based on real data)
# For now, let's assume feature values range between 0.5 and 2.5

np.random.seed(42)
synthetic_autism_sample = np.random.normal(loc=1.5, scale=0.3, size=(350, 16))  # shape = (350, 16)
synthetic_autism_tensor = torch.tensor(synthetic_autism_sample, dtype=torch.float32).unsqueeze(0)  # add batch dim

# Step 3: Predict
with torch.no_grad():
    output = model(synthetic_autism_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

print("🧠 Predicted class for synthetic autism EEG signal:", predicted_class)

# Step 4: Visualize the synthetic signal
plt.figure(figsize=(10, 4))
plt.plot(synthetic_autism_sample[:, 0], label='Feature 1 (example)')
plt.title("Synthetic Autism EEG Signal - Feature 1 over Time")
plt.xlabel("Timesteps")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
