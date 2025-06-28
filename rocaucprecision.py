import numpy as np
import torch
from sklearn.metrics import classification_report, roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ✅ Load Data (Training, Validation, Test sets)
X_train = np.load("train_X.npy")  # EEG signals for training
y_train = np.load("train_y.npy")  # Labels for training (0 = Normal, 1 = Autism)

X_val_features = np.load("val_X.npy")  # EEG signals for validation
y_val = np.load("val_y.npy")  # Labels for validation

X_test_features = np.load("test_X.npy")  # EEG signals for testing
y_test = np.load("test_y.npy")  # Labels for testing

# ✅ Define the EEG RNN Model (Architecture)
class EEG_RNN(torch.nn.Module):
    def __init__(self, input_size=16, hidden_size=128, num_layers=2, num_classes=2):
        super(EEG_RNN, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# ✅ Load Pre-trained Model (Make sure you have the trained model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEG_RNN().to(device)
model.load_state_dict(torch.load("eeg_rnn_best_model18april.pth", map_location=device))
model.eval()

# ✅ Prediction for the test set
def predict(model, X_data):
    model.eval()
    X_data_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(X_data_tensor)
        predicted = torch.argmax(output, dim=1).cpu().numpy()  # Get class predictions
    return predicted

# ✅ Make Predictions
y_pred = predict(model, X_test_features)

# If you have probabilities for ROC Curve (e.g., from model.predict_proba() in some cases)
# If your model outputs probabilities, get the probability for class 1 (Autism)
# Here we simulate it by using the softmax function on the model outputs to get class probabilities
with torch.no_grad():
    output = model(torch.tensor(X_test_features, dtype=torch.float32).to(device))
    softmax_output = torch.nn.functional.softmax(output, dim=1)
    y_proba = softmax_output[:, 1].cpu().numpy()  # Probabilities for class 1 (Autism)

# 📊 Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# ✅ Precision, Recall, F1-Score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-Score: {f1:.4f}")

# ✅ ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('🧠 ROC Curve - Autism EEG Detection')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
