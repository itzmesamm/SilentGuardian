import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("../Models/distress_model.h5")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save TFLite model
with open("../Models/distress_model_quantized.tflite", "wb") as f:
    f.write(tflite_model)

print("Quantized TFLite model saved successfully!")