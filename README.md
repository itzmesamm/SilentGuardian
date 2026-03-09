# SilentGuardian
SilentGuardian - Edge AI based autonomous emergency detection system using multimodal sensor fusion.
AI Backend Implementation (Current Progress)

The core AI modules for the SilentGuardian emergency detection system have been partially implemented. The system currently focuses on detecting distress situations using audio signals and smartphone motion sensor data.

Implemented Components

• Audio Distress Detection (CNN Model)
A Convolutional Neural Network (CNN) has been implemented to classify distress sounds using MFCC audio features extracted from the dataset.

• Audio Data Preprocessing
Audio signals are converted into MFCC feature representations to make them suitable for training the CNN model.

• Model Training and Evaluation
The CNN model was trained using labeled distress audio data and evaluated using metrics such as accuracy, confusion matrix, and classification report.

• Motion Detection (RNN Model)
A Recurrent Neural Network (RNN – LSTM based) has been implemented to analyze smartphone motion sensor patterns and detect abnormal movements.

• Motion Dataset Processing
Motion data from the UCI Human Activity Recognition (HAR) dataset was used to simulate smartphone accelerometer inputs.

• Model Inference Pipeline
An inference pipeline has been implemented to test both the CNN audio model and the RNN motion model using new sample inputs before integrating them into the mobile application.

• Mobile Deployment Preparation
The trained models have been converted to TensorFlow Lite format to enable efficient deployment on Android devices.

Datasets Used

Audio Distress Dataset
https://data.mendeley.com/datasets/gfvsdtnf3v/1

UCI Human Activity Recognition Dataset
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

Future Development (Remaining 50%)

• Integration of AI models into the Android application
• Real-time microphone distress detection
• Real-time accelerometer motion detection
• GPS-based emergency location tracking
• Multimodal fusion of audio, motion, and location signals
• Automatic emergency alert system for contacts and authorities
