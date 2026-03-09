import tensorflow as tf

print("Loading trained RNN model...")

model = tf.keras.models.load_model("../MotionModels/motion_rnn_model.h5")

print("Converting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Fix for LSTM conversion
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open("../MotionModels/motion_rnn_model.tflite", "wb") as f:
    f.write(tflite_model)

print("RNN model successfully converted to TFLite!")