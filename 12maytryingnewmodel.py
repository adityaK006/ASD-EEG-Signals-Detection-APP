# import os
# import random
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
# import seaborn as sns

# # ✅ Stability fix: Use legacy Keras backend (avoid TF NumPy bug)
# os.environ["TF_USE_LEGACY_KERAS"] = "1"

# # ✅ Fix randomness for reproducibility
# SEED = 42
# os.environ["PYTHONHASHSEED"] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# # ✅ Load preprocessed data
# X_train = np.load("train_X_balanced.npy")
# y_train = np.load("train_y_balanced.npy")
# X_val = np.load("val_X.npy")
# y_val = np.load("val_y.npy")
# X_test = np.load("test_X.npy")
# y_test = np.load("test_y.npy")

# # ✅ Cast data types
# X_train = X_train.astype("float32")
# X_val = X_val.astype("float32")
# X_test = X_test.astype("float32")

# y_train = y_train.astype("int64")
# y_val = y_val.astype("int64")
# y_test = y_test.astype("int64")

# print("✅ y_train unique values:", np.unique(y_train))
# print("✅ y_val unique values:", np.unique(y_val))
# print("✅ y_test unique values:", np.unique(y_test))

# # ✅ Define model architecture
# def build_eeg_rnn_model():
#     inputs = tf.keras.Input(shape=(350, 16), name="input_layer")
#     x = tf.keras.layers.LSTM(128, return_sequences=False, dropout=0.5, name="lstm")(inputs)
#     outputs = tf.keras.layers.Dense(2, activation="softmax", name="fc")(x)
#     model = tf.keras.Model(inputs, outputs)
#     return model

# model = build_eeg_rnn_model()
# model.summary()

# # ✅ Compile the model
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# # ✅ Early stopping
# early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# # ✅ Train the model
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=10,
#     batch_size=32,
#     callbacks=[early_stop]
# )

# # ✅ Evaluate on test data
# test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f"✅ Test Loss: {test_loss:.4f}")
# print(f"✅ Test Accuracy: {test_acc:.4f}")

# # ✅ Predict and evaluate manually
# y_probs = model.predict(X_test)
# y_pred = np.argmax(y_probs, axis=1)
# print("✅ Manual Accuracy:", accuracy_score(y_test, y_pred))

# # ✅ Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()

# # ✅ ROC Curve
# fpr, tpr, _ = roc_curve(y_test, y_probs[:, 1])
# roc_auc = roc_auc_score(y_test, y_probs[:, 1])

# plt.figure()
# plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='darkorange')
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()

# # ✅ Accuracy and Loss Plot
# plt.figure(figsize=(12, 4))

# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Train Acc')
# plt.plot(history.history['val_accuracy'], label='Val Acc')
# plt.title("Accuracy Over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.title("Loss Over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# # ✅ Save the model in SavedModel format
# model.export("eeg_rnn_tf2_model")  # For Keras 3

# # ✅ Convert to TensorFlow Lite
# converter = tf.lite.TFLiteConverter.from_saved_model("eeg_rnn_tf2_model")
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS,
#     tf.lite.OpsSet.SELECT_TF_OPS
# ]
# converter.experimental_enable_resource_variables = True
# converter._experimental_lower_tensor_list_ops = False

# tflite_model = converter.convert()

# # ✅ Save the TFLite model
# tflite_path = "eeg_rnn_tf2_model.tflite"
# with open(tflite_path, "wb") as f:
#     f.write(tflite_model)

# # ✅ Show TFLite model size
# model_size = os.path.getsize(tflite_path) / 1024 / 1024
# print(f"✅ LSTM model successfully converted to TFLite. Size: {model_size:.2f} MB")



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import seaborn as sns
import os
import random

# ✅ Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ✅ Load preprocessed data
X_train = np.load("train_X_balanced.npy")
y_train = np.load("train_y_balanced.npy")
X_val = np.load("val_X.npy")
y_val = np.load("val_y.npy")
X_test = np.load("test_X.npy")
y_test = np.load("test_y.npy")

# ✅ Cast data types
X_train = X_train.astype("float32")
X_val = X_val.astype("float32")
X_test = X_test.astype("float32")

y_train = y_train.astype("int32")
y_val = y_val.astype("int32")
y_test = y_test.astype("int32")

# ✅ Define 1D CNN model
def build_cnn_model():
    inputs = tf.keras.Input(shape=(350, 16), name="input")
    x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

model = build_cnn_model()
model.summary()

# ✅ Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ✅ Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ✅ Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
)

# ✅ Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Loss: {test_loss:.4f}")
print(f"✅ Test Accuracy: {test_acc:.4f}")

# ✅ Predictions and Confusion Matrix
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("✅ Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ✅ Accuracy Curve
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ✅ ROC AUC Curve
y_true_binary = tf.keras.utils.to_categorical(y_test, num_classes=2)
fpr, tpr, _ = roc_curve(y_true_binary[:, 1], y_pred_probs[:, 1])
auc = roc_auc_score(y_test, y_pred_probs[:, 1])

plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC AUC Curve")
plt.legend()
plt.grid(True)
plt.show()

# ✅ Save model
model.export("eeg_cnn_tf_model")

# ✅ Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("eeg_cnn_tf_model")
tflite_model = converter.convert()
with open("eeg_cnn_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ CNN model successfully converted to TFLite.")
