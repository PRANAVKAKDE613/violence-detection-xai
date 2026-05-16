import numpy as np
import tensorflow as tf
import cv2
import asyncio
from gradcam import get_gradcam_heatmap, overlay_heatmap, predict_and_explain
from telegram_alert import send_alert
import matplotlib.pyplot as plt

# Settings
MODEL_PATH = 'models/best_model.keras'
FRAMES_PER_VIDEO = 8
IMG_SIZE = 64
THRESHOLD = 0.5

def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, FRAMES_PER_VIDEO, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return np.array(frames)

def run_pipeline(video_path):
    print(f"\n--- Processing: {video_path} ---")
    
    # Step 1: Extract frames
    print("Step 1: Extracting frames...")
    frames = extract_frames_from_video(video_path)
    if len(frames) < FRAMES_PER_VIDEO:
        print("Not enough frames in video!")
        return
    
    # Step 2: Normalize
    frames_normalized = frames / 255.0
    
    # Step 3: Predict
    print("Step 2: Running prediction...")
    model = tf.keras.models.load_model(MODEL_PATH)
    input_tensor = np.expand_dims(frames_normalized, axis=0)
    prediction = model.predict(input_tensor, verbose=0)[0][0]
    label = 'VIOLENCE' if prediction > THRESHOLD else 'NO VIOLENCE'
    confidence = prediction if prediction > THRESHOLD else 1 - prediction
    print(f"Result: {label} ({confidence:.2%} confidence)")
    
    # Step 4: Generate Grad-CAM
    print("Step 3: Generating Grad-CAM heatmap...")
    heatmap, frame = get_gradcam_heatmap(model, frames_normalized)
    overlaid = overlay_heatmap(heatmap, frame)
    
    # Save heatmap image
    output_path = 'pipeline_output.png'
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow((frame * 255).astype(np.uint8))
    plt.title('Original Frame')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(overlaid)
    plt.title(f'{label} ({confidence:.2%})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Heatmap saved to {output_path}")
    
    # Step 5: Send Telegram alert if violence detected
    if label == 'VIOLENCE':
        print("Step 4: Sending Telegram alert...")
        asyncio.run(send_alert(output_path, label, confidence))
    else:
        print("Step 4: No violence detected — no alert sent.")
    
    print("--- Pipeline complete! ---\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        # Default test with a video from dataset
        test_video = 'data/violence/fi001.mp4'
        run_pipeline(test_video)
    else:
        run_pipeline(sys.argv[1])