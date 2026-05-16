import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def get_gradcam_heatmap(model, frames, layer_name='Conv_1'):
    # Get the MobileNetV2 inside TimeDistributed
    mobilenet = model.layers[1].layer
    
    # Build a model that outputs the conv layer + final prediction
    grad_model = tf.keras.models.Model(
        inputs=mobilenet.input,
        outputs=[mobilenet.get_layer(layer_name).output, mobilenet.output]
    )
    
    # Pick the middle frame
    frame = frames[len(frames)//2]
    frame_tensor = tf.cast(np.expand_dims(frame, axis=0), tf.float32)
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(frame_tensor)
        predicted_class = predictions[:, 0]
    
    # Compute gradients
    grads = tape.gradient(predicted_class, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Generate heatmap
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    
    return heatmap, frame

def overlay_heatmap(heatmap, frame, alpha=0.4):
    # Resize heatmap to frame size
    frame_uint8 = (frame * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay on frame
    overlaid = cv2.addWeighted(frame_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    return overlaid

def predict_and_explain(model, npy_path, save_path='gradcam_output.png'):
    # Load frames
    frames = np.load(npy_path)
    frames_normalized = frames / 255.0
    
    # Predict
    input_tensor = np.expand_dims(frames_normalized, axis=0)
    prediction = model.predict(input_tensor, verbose=0)[0][0]
    label = 'VIOLENCE' if prediction > 0.5 else 'NO VIOLENCE'
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    print(f"Prediction: {label} ({confidence:.2%} confidence)")
    
    # Generate Grad-CAM
    heatmap, frame = get_gradcam_heatmap(model, frames_normalized)
    overlaid = overlay_heatmap(heatmap, frame)
    
    # Save result
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow((frame * 255).astype(np.uint8))
    plt.title('Original Frame')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlaid)
    plt.title(f'Grad-CAM: {label} ({confidence:.2%})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    # Load model
    model = tf.keras.models.load_model('models/best_model.keras')
    
    # Test on one violence video
    predict_and_explain(
        model,
        npy_path='data/frames/violence/fi001.npy',
        save_path='gradcam_output.png'
    )