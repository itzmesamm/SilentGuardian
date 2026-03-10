from audio_inference import predict_audio
from motion_inference import predict_motion

print("SilentGuardian Emergency Monitoring Started")

audio_alert = predict_audio("test_audio1.wav")
motion_alert = predict_motion()

print("Evaluating emergency condition...")

if audio_alert and motion_alert:
    print("Emergency detected!")
    print("Triggering alert and sending location")
else:
    print("No emergency detected")