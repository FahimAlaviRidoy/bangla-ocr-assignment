import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from streamlit_drawable_canvas import st_canvas

# Configuration
MODEL_PATH = "models/model.keras"
LABELS_PATH = "labels.json"
IMG_SIZE = 32

@st.cache_resource
def load_ocr_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    return model, labels

def segment_characters(image):
    # Convert canvas image to grayscale and invert
    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours left-to-right
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours, bounding_boxes = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0]))
    
    chars = []
    for x, y, w, h in bounding_boxes:
        if w > 10 and h > 10: # Filter small noise
            char_img = thresh[y:y+h, x:x+w]
            # Pad to make it square before resizing
            padded = cv2.copyMakeBorder(char_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
            resized = cv2.resize(padded, (IMG_SIZE, IMG_SIZE))
            
            # Convert grayscale to 3-channel RGB (Required for Custom CNN / MobileNetV2)
            rgb_char = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            normalized = rgb_char.astype('float32') / 255.0
            
            chars.append(normalized)
            
    return np.array(chars), bounding_boxes

st.title("Bangla Handwritten Word Recognition")
st.write("Draw a simple Bangla word below. The system will segment it into characters and predict them.")

try:
    model, label_map = load_ocr_model()
except Exception as e:
    st.error("Model or labels not found. Please run `train.py` first.")
    st.stop()

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  
    stroke_width=8,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=200,
    width=600,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict Word"):
    if canvas_result.image_data is not None:
        chars, boxes = segment_characters(canvas_result.image_data)
        
        if len(chars) > 0:
            predictions = model.predict(chars)
            predicted_word_labels = []
            
            st.write("### Per-Character Predictions:")
            cols = st.columns(len(chars))
            
            for idx, pred in enumerate(predictions):
                class_idx = str(np.argmax(pred))
                confidence = np.max(pred)
                predicted_label = label_map[class_idx]
                predicted_word_labels.append(predicted_label)
                
                with cols[idx]:
                    # Display the character (just taking one channel for clean display)
                    st.image(chars[idx][:, :, 0], width=50)
                    st.write(f"**{predicted_label}**")
                    st.caption(f"{confidence:.2f}")
            
            st.success(f"### Final Predicted Sequence: {' '.join(predicted_word_labels)}")
        else:
            st.warning("No characters detected. Please draw clearer separated characters.")