## How to Run

### 1. Install dependencies
```bash
pip install tensorflow opencv-python albumentations scikit-learn python-telegram-bot matplotlib
```

### 2. Prepare dataset
```bash
python extract_frames.py
```

### 3. Train the model
```bash
python train.py
```

### 4. Run on a video file
```bash
python pipeline.py path/to/video.mp4
```

### 5. Run on webcam
```bash
python webcam.py
```

## Results
- Dataset: 300 surveillance video clips (150 fight + 150 non-fight)
- Model: MobileNetV2 (CNN) + LSTM
- Features: Real-time detection, Grad-CAM explainability, Telegram alerts

## Future Improvements
- Train on larger dataset (12,000+ clips) for higher accuracy
- Add GPU support for faster inference
- Integrate with CCTV camera systems
- Build web dashboard for monitoring
