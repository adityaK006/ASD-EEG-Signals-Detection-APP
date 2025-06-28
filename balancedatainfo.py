#Used this to balance the dataset autism and normal
import numpy as np
from sklearn.utils import resample, shuffle

# ✅ Step 0: Load the saved features and labels
X_train = np.load("train_X.npy")
y_train = np.load("train_y.npy")
X_val_features = np.load("val_X.npy")
y_val = np.load("val_y.npy")
X_test_features = np.load("test_X.npy")
y_test = np.load("test_y.npy")

# ✅ Step 1: Check original class distribution
unique, counts = np.unique(y_train, return_counts=True)
print("🔍 Original Training Class Distribution:", dict(zip(unique, counts)))

# ✅ Step 2: Split training data by class
X_train_autism = X_train[y_train == 1]  # Majority class
X_train_normal = X_train[y_train == 0]  # Minority class

y_train_autism = y_train[y_train == 1]
y_train_normal = y_train[y_train == 0]

# ✅ Step 3: Upsample the minority class (normal subjects)
X_normal_upsampled, y_normal_upsampled = resample(
    X_train_normal,
    y_train_normal,
    replace=True,                          # Sample with replacement
    n_samples=len(X_train_autism),         # Match the number of autism samples
    random_state=42
)

# ✅ Step 4: Combine the upsampled normal data with autism data
X_train_balanced = np.concatenate((X_train_autism, X_normal_upsampled), axis=0)
y_train_balanced = np.concatenate((y_train_autism, y_normal_upsampled), axis=0)

# ✅ Step 5: Shuffle the balanced data
X_train_balanced, y_train_balanced = shuffle(X_train_balanced, y_train_balanced, random_state=42)

# ✅ Step 6: Confirm new distribution
unique_balanced, counts_balanced = np.unique(y_train_balanced, return_counts=True)
print("✅ Balanced Training Class Distribution:", dict(zip(unique_balanced, counts_balanced)))
print("✅ Balanced Data Shape:", X_train_balanced.shape)

# ✅ Step 7: Save the new balanced training data
np.save("train_X_balanced.npy", X_train_balanced)
np.save("train_y_balanced.npy", y_train_balanced)

print("✅ Balanced datasets saved successfully.")
