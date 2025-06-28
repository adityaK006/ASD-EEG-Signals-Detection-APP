# #Working for previous model(overfit model)
# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import mne

# # ✅ Load Saved Data
# X_train = np.load("train_X_balanced.npy")  # EEG signals: (samples, time, channels)
# y_train = np.load("train_y_balanced.npy")  # Labels: (0 = Normal, 1 = Autism)

# # ✅ Define the EEG RNN Model (must match training)
# class EEG_RNN(nn.Module):
#     def __init__(self, input_size=16, hidden_size=128, num_layers=2, num_classes=2):
#         super(EEG_RNN, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         _, (hn, _) = self.lstm(x)
#         out = self.fc(hn[-1])
#         return out

# # ✅ Device Setup (GPU or CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"\n💻 Using device: {device}")

# # ✅ Load Trained Model
# model = EEG_RNN().to(device)
# model.load_state_dict(torch.load("eeg_rnn_best_model18april.pth", map_location=device))
# model.eval()

# # ✅ Function to Plot EEG Signal
# def plot_eeg(signal, title="EEG Signal", sample_rate=256):
#     plt.figure(figsize=(10, 4))
#     time_axis = np.arange(len(signal)) / sample_rate
#     plt.plot(time_axis, signal, label="EEG Signal", color="blue")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Amplitude")
#     plt.title(title)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# # ✅ Visualize First Autism and Normal Sample
# autism_index = np.where(y_train == 1)[0][0]
# normal_index = np.where(y_train == 0)[0][0]

# autism_signal = X_train[autism_index, :, 0]
# normal_signal = X_train[normal_index, :, 0]

# # plot_eeg(autism_signal, title="Autism EEG Signal (Sample)")
# # plot_eeg(normal_signal, title="Normal EEG Signal (Sample)")

# # ✅ Random Sample: Actual vs Predicted
# random_idx = np.random.randint(0, X_train.shape[0])
# random_signal = X_train[random_idx]
# actual_label = y_train[random_idx]  # 0 = Normal, 1 = Autism

# # ✅ Prepare input tensor
# input_tensor = torch.tensor(random_signal, dtype=torch.float32).unsqueeze(0).to(device)

# # ✅ Predict
# with torch.no_grad():
#     output = model(input_tensor)
#     predicted_class = torch.argmax(output, dim=1).item()

# # ✅ Show Results
# print(f"\n🎯 Actual Label   : {'Autism' if actual_label == 1 else 'Normal'}")
# print(f"🤖 Predicted Label: {'Autism' if predicted_class == 1 else 'Normal'}")

# # ✅ Plot EEG Signal with Actual and Predicted Labels
# plot_eeg(random_signal[:, 0], title=f"Actual Signal - Label: {'Autism' if actual_label == 1 else 'Normal'}")
# plot_eeg(random_signal[:, 0], title=f"Predicted Signal - Label: {'Autism' if predicted_class == 1 else 'Normal'}")




#May 09 with latest tf model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ✅ Load Saved Data
X_train = np.load("train_X_balanced.npy")  # Shape: (samples, time, channels)
y_train = np.load("train_y_balanced.npy")  # 0 = Normal, 1 = Autism

# ✅ Load TFLite Model
interpreter = tf.lite.Interpreter(model_path="eeg_rnn_tf_model.tflite")
interpreter.allocate_tensors()

# ✅ Get Input/Output Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Function to Plot EEG Signal
def plot_eeg(signal, title="EEG Signal", sample_rate=256):
    plt.figure(figsize=(10, 4))
    time_axis = np.arange(len(signal)) / sample_rate
    plt.plot(time_axis, signal, label="EEG Signal", color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ✅ Run Inference on a Random Sample
random_idx = np.random.randint(0, X_train.shape[0])
random_signal = X_train[random_idx]  # Shape: (time, channels)
actual_label = y_train[random_idx]

# ✅ Prepare Input (reshape to match TFLite: (1, time, channels))
input_tensor = np.expand_dims(random_signal, axis=0).astype(np.float32)

# ✅ Run Inference
interpreter.set_tensor(input_details[0]['index'], input_tensor)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

predicted_class = np.argmax(output)

# ✅ Print Results
print(f"\n🎯 Actual Label   : {'Autism' if actual_label == 1 else 'Normal'}")
print(f"🤖 Predicted Label: {'Autism' if predicted_class == 1 else 'Normal'}")

# ✅ Plot EEG Signal
plot_eeg(random_signal[:, 0], title=f"EEG Signal — Actual: {'Autism' if actual_label == 1 else 'Normal'} | Predicted: {'Autism' if predicted_class == 1 else 'Normal'}")
