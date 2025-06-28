import tensorflow as tf

# Load your fully weighted model (Replace with actual code to load your model)
from numpytotensorflow import build_tf_model  # Replace with your actual model file if needed

model = build_tf_model()

# Ensure the model has static input shapes
model.build(input_shape=(None, 350, 16))  # Example static shape; adjust as necessary

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Disable tensor list lowering
converter._experimental_lower_tensor_list_ops = False

# Use SELECT_TF_OPS for more operations compatibility
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

# Optional: Apply optimizations for model size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
try:
    tflite_model = converter.convert()

    # Save the converted model
    with open("eeg_rnn_model.tflite", "wb") as f:
        f.write(tflite_model)

    print("✅ Model successfully converted to TensorFlow Lite: eeg_rnn_model.tflite")

except Exception as e:
    print(f"❌ Error during conversion: {e}")
