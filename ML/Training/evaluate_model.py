import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# Load trained model
model = tf.keras.models.load_model("../Models/distress_model.h5")

# Load test dataset
X_test = np.load("../Models/X_test.npy")
y_test = np.load("../Models/y_test.npy")

# Add channel dimension
X_test = X_test[..., np.newaxis]

# Predict
predictions = model.predict(X_test)

y_pred = (predictions > 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
