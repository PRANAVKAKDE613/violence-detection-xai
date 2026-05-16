import cv2
import numpy as np
import tensorflow as tf
import asyncio
import threading
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gradcam import get_gradcam_heatmap, overlay_heatmap
from telegram_alert import send_alert

# Settings
MODEL_PATH = 'models/best_model.keras'
FRAMES_PER_VIDEO = 8
IMG_SIZE = 64
THRESHOLD = 0.5
ALERT_COOLDOWN = 30

# Global
last_alert_time = 0

def send_alert_async(image_path, label, confidence):
    asyncio.run(send_alert(image_path, label, confidence))

def connect_camera(source):
    """
    source can be:
    - 0 = webcam
    - 'rtsp://username:password@ip_address:554/stream' = CCTV
    - 'http://ip_address:port/video' = IP camera
    - 'path/to/video.mp4' = video file
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot connect to: {source}")
        return None
    print(f"Connected to: {source}")
    return cap

def run_cctv(source=0, camera_name="Camera 1"):
    global last_alert_time

    print(f"\nLoading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded!")

    cap = connect_camera(source)
    if cap is None:
        return

    frame_buffer = []
    print(f"Monitoring {camera_name}... Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection — retrying...")
            time.sleep(2)
            cap = connect_camera(source)
            if cap is None:
                break
            continue

        # Show feed with camera name
        display = frame.copy()
        cv2.putText(display, camera_name, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Collect frames
        small = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        frame_buffer.append(small_rgb)

        if len(frame_buffer) == FRAMES_PER_VIDEO:
            frames_array = np.array(frame_buffer) / 255.0
            input_tensor = np.expand_dims(frames_array, axis=0)
            prediction = model.predict(input_tensor, verbose=0)[0][0]
            label = 'VIOLENCE' if prediction > THRESHOLD else 'SAFE'
            confidence = float(prediction if prediction > THRESHOLD else 1 - prediction)

            # Show label on frame
            color = (0, 0, 255) if label == 'VIOLENCE' else (0, 255, 0)
            cv2.putText(display, f"{label} {confidence:.2%}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                       1.2, color, 3)

            print(f"[{camera_name}] {label} — {confidence:.2%}")

            # Send alert if violence
            current_time = time.time()
            if label == 'VIOLENCE' and (current_time - last_alert_time) > ALERT_COOLDOWN:
                last_alert_time = current_time
                print(f"Sending alert for {camera_name}...")

                heatmap, mid_frame = get_gradcam_heatmap(model, frames_array)
                overlaid = overlay_heatmap(heatmap, mid_frame)

                output_path = f'alert_{camera_name.replace(" ", "_")}.png'
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.imshow((mid_frame * 255).astype(np.uint8))
                plt.title(f'{camera_name} — Original')
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(overlaid)
                plt.title(f'{label} ({confidence:.2%})')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()

                t = threading.Thread(
                    target=send_alert_async,
                    args=(output_path, f"[{camera_name}] {label}", confidence)
                )
                t.start()

            frame_buffer = []

        cv2.imshow(f'Violence Detection — {camera_name}', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Select camera source:")
    print("1. Webcam")
    print("2. RTSP Camera (CCTV)")
    print("3. IP Camera")
    print("4. Video File")
    choice = input("Enter choice (1/2/3/4): ")

    if choice == '1':
        run_cctv(source=0, camera_name="Webcam")

    elif choice == '2':
        ip = input("Enter RTSP URL (e.g. rtsp://admin:password@192.168.1.100:554/stream): ")
        run_cctv(source=ip, camera_name="CCTV Camera")

    elif choice == '3':
        ip = input("Enter IP camera URL (e.g. http://192.168.1.100:8080/video): ")
        run_cctv(source=ip, camera_name="IP Camera")

    elif choice == '4':
        path = input("Enter video file path: ")
        run_cctv(source=path, camera_name="Video File")