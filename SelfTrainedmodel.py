from ultralytics import YOLO
import cv2
import torch

# Load the trained model
model_path = r"C:\Users\gaurm\OneDrive\Desktop\YoLO\YOLO_helmet_training\helmet_detection\weights\best.pt"
model = YOLO(model_path)

# Check if CUDA is available and use GPU if possible
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Model loaded successfully!")

# Load an image
image_path = r"C:\Users\gaurm\Downloads\IMG_5783.jpg"

# Run inference
results = model(image_path)  # Model inference

# Display results
results[0].show()  



