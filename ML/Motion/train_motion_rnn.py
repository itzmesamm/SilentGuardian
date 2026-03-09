import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print("Loading dataset...")

X = np.load("../MotionModels/X_motion.npy")
y = np.load("../MotionModels/y_motion.npy")

X = X.reshape((X.shape[0],X.shape[1],1))

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2
)

model = tf.keras.Sequential([

    layers.LSTM(64,return_sequences=True,input_shape=(X.shape[1],1)),
    layers.Dropout(0.3),

    layers.LSTM(32),

    layers.Dense(32,activation="relu"),

    layers.Dense(1,activation="sigmoid")

])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test,y_test)
)

model.save("../MotionModels/motion_rnn_model.h5")

print("Motion RNN model saved")