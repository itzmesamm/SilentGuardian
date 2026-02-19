import os
import numpy as np
import librosa

# =============================
# Configuration
# =============================

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 128

DATA_PATH = "../Data"   # Using YOUR folder name


# =============================
# Feature Extraction
# =============================

def extract_features(file_path):
    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=N_MFCC
        )

        # Pad or trim to fixed size
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]

        return mfcc

    except Exception as e:
        print(f"⚠ Skipping corrupted file: {file_path}")
        return None


# =============================
# Load Train/Test Split
# =============================

def load_split(split="train"):
    X = []
    y = []
    skipped_files = 0

    split_path = os.path.join(DATA_PATH, split)

    print(f"\nLoading {split} data...")

    for label_folder in os.listdir(split_path):
        folder_path = os.path.join(split_path, label_folder)

        # Merge Child + Women → distress (1)
        # Normal → 0
        if label_folder.lower() == "normal":
            label = 0
        else:
            label = 1

        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)

                features = extract_features(file_path)

                if features is not None:
                    X.append(features)
                    y.append(label)
                else:
                    skipped_files += 1

    print(f"{split.capitalize()} loaded.")
    print(f"Skipped corrupted files: {skipped_files}")

    return np.array(X), np.array(y)


# =============================
# Main Execution
# =============================

if __name__ == "__main__":

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    print("\n=============================")
    print("Final Dataset Summary")
    print("=============================")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Train label distribution:", np.bincount(y_train))
    print("Test label distribution:", np.bincount(y_test))

    # Save processed data
    np.save("../Models/X_train.npy", X_train)
    np.save("../Models/y_train.npy", y_train)
    np.save("../Models/X_test.npy", X_test)
    np.save("../Models/y_test.npy", y_test)

    print("\nSaved processed arrays inside Models folder.")
