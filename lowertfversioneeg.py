import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

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

# ✅ Define model architecture
def build_eeg_rnn_model():
    inputs = tf.keras.Input(shape=(350, 16), name="input_layer")
    x = tf.keras.layers.LSTM(128, return_sequences=False, dropout=0.5, name="lstm")(inputs)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="fc")(x)
    model = tf.keras.Model(inputs, outputs)
    return model

model = build_eeg_rnn_model()
model.summary()

# ✅ Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy"
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
# Evaluate only the loss
test_loss = model.evaluate(X_test, y_test)
print(f"✅ Test Loss: {test_loss:.4f}")

# Predict class labels
y_pred = np.argmax(model.predict(X_test), axis=1)

# Manually compute accuracy
test_acc = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy (manual): {test_acc:.4f}")
print(f"✅ Test Accuracy: {test_acc:.4f}")

# ✅ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Manual accuracy:", accuracy_score(y_test, y_pred))

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ✅ Save the model
# ✅ Export the model in SavedModel format for TFLite conversion
model.export("eeg_rnn_tf_model")  # This is the correct method in Keras 3

# ✅ Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model("eeg_rnn_tf_model")

# ✅ Fix 1: Enable support for Select TF Ops (needed for LSTM)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,      # Basic TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS         # Enable TF ops like LSTM
]

# ✅ Fix 2: Allow resource variables (for LSTM internal states)
converter.experimental_enable_resource_variables = True

# ✅ Fix 3: Disable lowering of tensor list ops (prevents conversion crash)
converter._experimental_lower_tensor_list_ops = False

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open("eeg_rnn_tf_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ LSTM model successfully converted to TFLite.")







# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))