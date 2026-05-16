import cv2
import os
import numpy as np

# Settings
DATASET = {
    'violence': 'data/violence',
    'non_violence': 'data/non_violence'
}
OUTPUT_DIR = 'data/frames'
FRAMES_PER_VIDEO = 8
IMG_SIZE = 64

def extract_frames(video_path, num_frames=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)
    cap.release()
    return frames

for label, folder in DATASET.items():
    out_folder = os.path.join(OUTPUT_DIR, label)
    os.makedirs(out_folder, exist_ok=True)
    videos = os.listdir(folder)
    print(f"Processing {label}: {len(videos)} videos")
    for vid in videos:
        vid_path = os.path.join(folder, vid)
        frames = extract_frames(vid_path)
        vid_name = os.path.splitext(vid)[0]
        save_path = os.path.join(out_folder, vid_name + '.npy')
        np.save(save_path, np.array(frames))
    print(f"Done: {label}")

print("All frames extracted!")