import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils import class_weight

# =============================
# Paths
# =============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, "..", "Models")

# =============================
# Load Dataset
# =============================

print("Loading processed dataset...")

X_train = np.load(os.path.join(MODELS_PATH, "X_train.npy"))
y_train = np.load(os.path.join(MODELS_PATH, "y_train.npy"))

X_test = np.load(os.path.join(MODELS_PATH, "X_test.npy"))
y_test = np.load(os.path.join(MODELS_PATH, "y_test.npy"))

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =============================
# Reshape for CNN
# =============================

# CNN expects 4D input
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("Reshaped train:", X_train.shape)

# =============================
# Handle Class Imbalance
# =============================

weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(weights))

print("Class weights:", class_weights)

# =============================
# Build CNN Model
# =============================

model = models.Sequential([

    layers.Conv2D(32, (3,3), activation='relu', input_shape=(40,128,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(1, activation='sigmoid')   # Binary classification
])

# =============================
# Compile Model
# =============================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:\n")
model.summary()

# =============================
# Train Model
# =============================

print("\nStarting training...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights
)

# =============================
# Evaluate Model
# =============================

print("\nEvaluating model...\n")

test_loss, test_acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_acc)

# =============================
# Save Model
# =============================

model_path = os.path.join(MODELS_PATH, "distress_model.h5")

model.save(model_path)

print("\nModel saved at:", model_path)