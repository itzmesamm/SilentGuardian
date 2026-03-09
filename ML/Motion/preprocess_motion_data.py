import numpy as np
import os

DATA_PATH = "../MotionData/UCI HAR Dataset/train"

print("Loading motion dataset...")

X = np.loadtxt(os.path.join(DATA_PATH,"X_train.txt"))
y = np.loadtxt(os.path.join(DATA_PATH,"y_train.txt"))

# Convert labels to binary (0 = normal)
y_binary = np.zeros_like(y)

np.save("../MotionModels/X_motion.npy",X)
np.save("../MotionModels/y_motion.npy",y_binary)

print("Motion dataset prepared")
print("Dataset shape:",X.shape)