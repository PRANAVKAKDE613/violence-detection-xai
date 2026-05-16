from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import tensorflow as tf
import asyncio
import threading
import base64
import time
from gradcam import get_gradcam_heatmap, overlay_heatmap
from telegram_alert import send_alert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Settings
MODEL_PATH = 'models/best_model.keras'
FRAMES_PER_VIDEO = 8
IMG_SIZE = 64
THRESHOLD = 0.5
ALERT_COOLDOWN = 30

# Global state
model = None
camera_running = False
last_alert_time = 0
detection_log = []

def load_model():
    global model
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded!")

def send_alert_async(image_path, label, confidence):
    asyncio.run(send_alert(image_path, label, confidence))

def camera_thread():
    global camera_running, last_alert_time, detection_log

    cap = cv2.VideoCapture(0)
    frame_buffer = []

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        # Encode frame to base64 and send to browser
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # Collect frames
        small_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(small_frame_rgb)

        if len(frame_buffer) == FRAMES_PER_VIDEO:
            frames_array = np.array(frame_buffer) / 255.0
            input_tensor = np.expand_dims(frames_array, axis=0)
            prediction = model.predict(input_tensor, verbose=0)[0][0]
            label = 'VIOLENCE' if prediction > THRESHOLD else 'SAFE'
            confidence = float(prediction if prediction > THRESHOLD else 1 - prediction)

            # Log detection
            log_entry = {
                'time': time.strftime('%H:%M:%S'),
                'label': label,
                'confidence': f"{confidence:.2%}"
            }
            detection_log.insert(0, log_entry)
            detection_log = detection_log[:20]  # keep last 20

            # Send to browser via socketio
            socketio.emit('detection', {
                'frame': frame_b64,
                'label': label,
                'confidence': f"{confidence:.2%}",
                'log': detection_log
            })

            # Telegram alert
            current_time = time.time()
            if label == 'VIOLENCE' and (current_time - last_alert_time) > ALERT_COOLDOWN:
                last_alert_time = current_time

                heatmap, mid_frame = get_gradcam_heatmap(model, frames_array)
                overlaid = overlay_heatmap(heatmap, mid_frame)

                output_path = 'static/alert.png'
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

                t = threading.Thread(
                    target=send_alert_async,
                    args=(output_path, label, confidence)
                )
                t.start()

                socketio.emit('alert', {
                    'time': log_entry['time'],
                    'confidence': f"{confidence:.2%}"
                })

            frame_buffer = []

        time.sleep(0.01)

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_camera():
    global camera_running
    if not camera_running:
        camera_running = True
        t = threading.Thread(target=camera_thread)
        t.daemon = True
        t.start()
    return jsonify({'status': 'started'})

@app.route('/stop', methods=['POST'])
def stop_camera():
    global camera_running
    camera_running = False
    return jsonify({'status': 'stopped'})

@app.route('/log')
def get_log():
    return jsonify(detection_log)

if __name__ == '__main__':
    load_model()
    import os
    os.makedirs('static', exist_ok=True)
    print("Starting dashboard at http://localhost:5000")
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)