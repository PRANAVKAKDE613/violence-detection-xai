import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_model

# Load data
def load_data():
    X, y = [], []
    for label, folder in [('violence', 1), ('non_violence', 0)]:
        path = f'data/frames/{label}'
        files = os.listdir(path)
        print(f"Loading {label}: {len(files)} files")
        for f in files:
            frames = np.load(os.path.join(path, f))
            if frames.shape[0] == 8:
                X.append(frames)
                y.append(1 if label == 'violence' else 0)
    return np.array(X), np.array(y)

print("Loading data...")
X, y = load_data()

# Normalize pixels to 0-1
X = X / 255.0

print(f"Dataset shape: {X.shape}")
print(f"Labels: {np.unique(y, return_counts=True)}")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Build model
model = build_model()

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint = ModelCheckpoint(
    'models/best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    verbose=1
)

# Train
print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=8,
    callbacks=[checkpoint, early_stop]
)

print("Training complete!")
print(f"Best val accuracy: {max(history.history['val_accuracy']):.2%}")