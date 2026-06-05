import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.tensorflow

# Configuration
DATASET_PATH = "./dataset" 
MODEL_SAVE_PATH = "models/model.keras"
LABELS_SAVE_PATH = "labels.json"
IMG_SIZE = 32 # MobileNetV2 supports 32x32 minimum

def load_data(dataset_path):
    images = []
    labels = []
    label_map = {}
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} not found.")

    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for idx, class_name in enumerate(sorted(class_dirs)):
        label_map[idx] = class_name
        class_dir = os.path.join(dataset_path, class_name)
        
        for img_name in os.listdir(class_dir)[:500]: 
            img_path = os.path.join(class_dir, img_name)
            # Load in color (3 channels) because MobileNetV2 requires it
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(idx)
                
    # Normalize images
    images = np.array(images).astype('float32') / 255.0
    labels = np.array(labels)
    return images, labels, label_map

def build_cnn_model(num_classes):
    """Custom Convolutional Neural Network"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_mobilenet_model(num_classes):
    """Transfer Learning model using MobileNetV2"""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    # Freeze the base model to speed up training and prevent overfitting
    base_model.trainable = False 
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("Loading data (this may take a moment)...")
    X, y, label_map = load_data(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with open(LABELS_SAVE_PATH, "w") as f:
        json.dump(label_map, f)
        
    num_classes = len(label_map)
    mlflow.set_experiment("Bangla_OCR_Experiment")
    
    # Define our two different model architectures for the experiment
    experiment_runs = [
        {"run_name": "Model_1_Custom_CNN", "type": "cnn", "epochs": 5, "batch_size": 64},
        {"run_name": "Model_2_MobileNetV2", "type": "mobilenet", "epochs": 5, "batch_size": 64}
    ]
    
    best_val_acc = 0.0
    best_model = None

    for exp in experiment_runs:
        with mlflow.start_run(run_name=exp["run_name"]):
            print(f"\n--- Starting {exp['run_name']} ---")
            
            # Log parameters
            mlflow.log_param("model_type", exp["type"])
            mlflow.log_param("epochs", exp["epochs"])
            mlflow.log_param("batch_size", exp["batch_size"])
            mlflow.log_param("img_size", IMG_SIZE)
            
            # Build the specific model
            if exp["type"] == "cnn":
                model = build_cnn_model(num_classes)
            else:
                model = build_mobilenet_model(num_classes)
                
            history = model.fit(X_train, y_train, 
                                epochs=exp["epochs"], 
                                validation_data=(X_test, y_test), 
                                batch_size=exp["batch_size"])
            
            # Log metrics
            val_acc = history.history['val_accuracy'][-1]
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.tensorflow.log_model(model, "model")
            
            # Track the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                print(f"*** New best model found! ({exp['run_name']}) Validation Accuracy: {best_val_acc:.4f} ***")

    # Save ONLY the overall best model for the app to use
    if best_model:
        os.makedirs("models", exist_ok=True)
        best_model.save(MODEL_SAVE_PATH)
        print(f"\nSuccessfully saved the best model to {MODEL_SAVE_PATH} with accuracy {best_val_acc:.4f}")

if __name__ == "__main__":
    main()