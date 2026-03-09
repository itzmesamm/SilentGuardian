import numpy as np
import tensorflow as tf

MODEL_PATH = "../MotionModels/motion_rnn_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

def predict_motion():

    sample_motion = np.random.rand(1,561,1)

    prediction = model.predict(sample_motion)[0][0]

    if prediction > 0.5:
        print("Fall or abnormal motion detected")
        return True
    else:
        print("Normal motion")
        return False


if __name__ == "__main__":
    predict_motion()