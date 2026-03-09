import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix,classification_report

X = np.load("../MotionModels/X_motion.npy")
y = np.load("../MotionModels/y_motion.npy")

X = X.reshape((X.shape[0],X.shape[1],1))

model = tf.keras.models.load_model("../MotionModels/motion_rnn_model.h5")

pred = model.predict(X)

pred = (pred>0.5).astype(int)

print("Confusion Matrix")
print(confusion_matrix(y,pred))

print("\nClassification Report")
print(classification_report(y,pred))