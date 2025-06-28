import torch
import numpy as np
from model import EEG_RNN  # Replace with your actual model definition

# Load the trained PyTorch model
pytorch_model = EEG_RNN()
pytorch_model.load_state_dict(torch.load("eeg_rnn_best_model18april.pth", map_location=torch.device('cpu')))
pytorch_model.eval()

# Extract weights from LSTM and fully connected layer
lstm_weights = {}
for name, param in pytorch_model.lstm.named_parameters():
    lstm_weights[name] = param.detach().numpy()

fc_weights = {
    "weight": pytorch_model.fc.weight.detach().numpy(),
    "bias": pytorch_model.fc.bias.detach().numpy()
}

# ✅ Save these NumPy weights to disk if needed
np.savez("lstm_weights.npz", **lstm_weights)
np.savez("fc_weights.npz", **fc_weights)

print("✅ PyTorch LSTM and FC weights extracted and saved as NumPy arrays.")
