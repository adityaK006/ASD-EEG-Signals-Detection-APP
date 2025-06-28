import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ Using CPU — GPU not available")

# ✅ Define EEG Dataset
class EEGDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ✅ Define the RNN Model with Dropout
class EEG_RNN(nn.Module):
    def __init__(self, input_size=16, hidden_size=128, num_layers=2, num_classes=2):
        super(EEG_RNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)  # Dropout added
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# ✅ Load preprocessed data
X_train = np.load("train_X_balanced.npy")
y_train = np.load("train_y_balanced.npy")
X_val = np.load("val_X.npy")
y_val = np.load("val_y.npy")
X_test = np.load("test_X.npy")
y_test = np.load("test_y.npy")

# ✅ Create DataLoaders
train_dataset = EEGDataset(X_train, y_train)
val_dataset = EEGDataset(X_val, y_val)
test_dataset = EEGDataset(X_test, y_test)

batch_size = 32
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ✅ Initialize Model, Loss, Optimizer

model = EEG_RNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # L2 Regularization

# ✅ Training with Early Stopping
num_epochs = 15
patience = 3  # Early stopping patience
best_val_loss = float("inf")
patience_counter = 0

train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    total_loss, correct = 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == y_batch).sum().item()

    train_acc = correct / len(train_dataset)
    train_losses.append(total_loss)

    # ✅ Validate Model
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == y_batch).sum().item()

    val_acc = val_correct / len(val_dataset)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # ✅ Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "eeg_rnn_best_model18april.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("🛑 Early stopping triggered")
            break

print("✅ Model training complete and saved!")

# ✅ Plot Loss vs Epochs
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# ✅ Test Model Performance
model.load_state_dict(torch.load("eeg_rnn_best_model18april.pth"))  # Load best model
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        predictions = outputs.argmax(1).cpu().numpy()
        y_true.extend(y_batch.cpu().numpy())
        y_pred.extend(predictions)

test_acc = accuracy_score(y_true, y_pred)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# ✅ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
