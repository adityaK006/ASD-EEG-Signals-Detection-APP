import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from model import EEG_RNN  # ✅ Replace with your actual model class file

# ✅ Load processed features and labels
X = np.load("train_X_balanced.npy")     # Shape: (146, 350, 16)
y = np.load("train_y_balanced.npy")     # Shape: (146,)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# ✅ Setup K-Fold
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# ✅ Evaluate using K-Fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_tensor)):
    print(f"\n🔁 Fold {fold + 1} / {k}")

    # Extract validation split
    X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

    # ✅ Load the saved trained model
    model = EEG_RNN()  # Initialize architecture
    model.load_state_dict(torch.load("eeg_rnn_best_model18april.pth"))  # Load trained weights
    model.eval()

    # ✅ Create dataloader for validation
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # ✅ Evaluate the model
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # ✅ Show metrics
    print(classification_report(all_labels, all_preds, digits=4))
