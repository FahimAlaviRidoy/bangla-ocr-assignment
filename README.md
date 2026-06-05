# Bangla Handwritten Word Recognition System

## Overview
This project builds a Bangla OCR system using the BanglaLekha-Isolated dataset. It includes automated model training via MLflow (comparing a Custom CNN against MobileNetV2), a Streamlit UI for user interaction, and Docker containerization.

## Technical Approach
1. **Preprocessing**: Images are loaded in 3-channel format (RGB), resized to 32x32 pixels, and normalized. This 3-channel requirement ensures compatibility with transfer learning architectures.
2. **Model Architectures & MLflow Tracking**: The training script is configured to track two competing architectures:
   * **Model 1 (Custom CNN)**: Built from scratch with 2 Conv2D layers, Max Pooling, Dropout, and a Dense softmax output layer.
   * **Model 2 (MobileNetV2)**: Utilizes Transfer Learning using pre-trained ImageNet weights, appended with a Global Average Pooling and Dropout layer. 
   MLflow tracks both runs, logging their parameters and validation accuracy. The script automatically evaluates which architecture achieves the highest accuracy and saves *only* the winning model.
3. **Word Segmentation Strategy**: The Streamlit canvas generates an RGBA image. It is converted to grayscale, thresholded, and `cv2.findContours` detects individual drawn strokes. Bounding boxes are sorted left-to-right. Finally, the extracted characters are converted back into a 3-channel tensor to match the required input shape of the trained models.

## How to Run

### 1. Local Setup
1. Extract the BanglaLekha-Isolated dataset into a folder named `dataset/` in the root directory.
2. Install dependencies: `pip install -r requirements.txt`

### 2. Training & MLflow
1. Start MLflow server: `mlflow ui --host 0.0.0.0 --port 5000` (View dashboard at http://localhost:5000)
2. In a new terminal, run training: `python train.py`
3. The script will train both the Custom CNN and MobileNetV2, log them to MLflow, and automatically save the superior model to `models/model.keras` along with the `labels.json` map.

### 3. Run Streamlit App
Run locally: `streamlit run app.py`

### 4. Docker
Build the image: `docker build -t bangla-ocr-app:0.1 .`
Run the container: `docker run -p 8501:8501 bangla-ocr-app:0.1`
Access at http://localhost:8501