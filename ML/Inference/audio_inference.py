import numpy as np
import tensorflow as tf
import librosa

MODEL_PATH = "../Models/distress_model.h5"
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 128

model = tf.keras.models.load_model(MODEL_PATH)

def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC
    )

    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc

def predict_audio(file_path):

    features = extract_features(file_path)

    features = features[np.newaxis, ..., np.newaxis]

    prediction = model.predict(features)[0][0]

    if prediction > 0.5:
        print("Distress sound detected")
        return True
    else:
        print("Normal audio")
        return False


if __name__ == "__main__":
    predict_audio("test_audio2.wav")