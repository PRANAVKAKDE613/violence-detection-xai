import cv2
import numpy as np
import tensorflow as tf
import asyncio
import threading
from gradcam import get_gradcam_heatmap, overlay_heatmap
from telegram_alert import send_alert
import matplotlib.pyplot as plt

# Settings
MODEL_PATH = 'models/best_model.keras'
FRAMES_PER_VIDEO = 8
IMG_SIZE = 64
THRESHOLD = 0.5
ALERT_COOLDOWN = 30  # seconds between alerts

# Load model once
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded! Starting webcam...")

# Alert cooldown tracker
last_alert_time = 0

def collect_frames(cap):
    frames = []
    total_frames = 8
    for _ in range(total_frames):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    return np.array(frames)

def send_alert_async(image_path, label, confidence):
    asyncio.run(send_alert(image_path, label, confidence))

def run_webcam():
    global last_alert_time
    
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    
    if not cap.isOpened():
        print("Cannot open webcam!")
        return
    
    print("Webcam started! Press Q to quit.")
    frame_buffer = []
    import time
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show live feed
        display_frame = frame.copy()
        
        # Collect frames into buffer
        small_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(small_frame_rgb)
        
        # Once we have enough frames run prediction
        if len(frame_buffer) == FRAMES_PER_VIDEO:
            frames_array = np.array(frame_buffer) / 255.0
            
            # Predict
            input_tensor = np.expand_dims(frames_array, axis=0)
            prediction = model.predict(input_tensor, verbose=0)[0][0]
            label = 'VIOLENCE' if prediction > THRESHOLD else 'SAFE'
            confidence = prediction if prediction > THRESHOLD else 1 - prediction
            
            # Show label on screen
            color = (0, 0, 255) if label == 'VIOLENCE' else (0, 255, 0)
            cv2.putText(
                display_frame,
                f"{label} ({confidence:.2%})",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, color, 3
            )
            
            # Send Telegram alert if violence and cooldown passed
            current_time = time.time()
            if label == 'VIOLENCE' and (current_time - last_alert_time) > ALERT_COOLDOWN:
                last_alert_time = current_time
                print(f"Violence detected! Sending alert...")
                
                # Generate Grad-CAM
                heatmap, mid_frame = get_gradcam_heatmap(model, frames_array)
                overlaid = overlay_heatmap(heatmap, mid_frame)
                
                # Save heatmap
                output_path = 'webcam_alert.png'
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.imshow((mid_frame * 255).astype(np.uint8))
                plt.title('Captured Frame')
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(overlaid)
                plt.title(f'{label} ({confidence:.2%})')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                
                # Send in background thread so webcam doesn't freeze
                t = threading.Thread(
                    target=send_alert_async,
                    args=(output_path, label, confidence)
                )
                t.start()
            
            # Reset buffer
            frame_buffer = []
        
        # Show live feed
        cv2.imshow('Violence Detection System', display_frame)
        
        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stopped.")

if __name__ == "__main__":
    run_webcam()