import tensorflow as tf

print("Loading trained RNN model...")

# Load trained model
model = tf.keras.models.load_model("../MotionModels/motion_rnn_model.h5")

print("Converting RNN model to TensorFlow Lite with quantization...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Fix for LSTM / RNN conversion
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter._experimental_lower_tensor_list_ops = False

# Convert model
tflite_model = converter.convert()

# Save quantized model
with open("../MotionModels/motion_rnn_model_quantized.tflite", "wb") as f:
    f.write(tflite_model)

print("Quantized RNN model successfully converted!")