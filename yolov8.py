import torch
import os
from ultralytics import YOLO

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def train_yolo():
    # Define paths
    data_yaml = r"C:\Users\gaurm\OneDrive\Desktop\data.yaml"
    model_checkpoint = r"C:\Users\gaurm\OneDrive\Desktop\YoLO\yolov8n.pt"

    # Initialize the YOLO model
    model = YOLO(model_checkpoint)

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=50,
        batch=16,
        imgsz=640,
        device=device,
        workers=4,  # Reduce to 0 if issues persist
        project="YOLO_helmet_training",
        name="helmet_detection",
        exist_ok=True,
    )

    print("Training completed successfully!")

    # Path where the best trained model is saved
    trained_model_path = r"YOLO_helmet_training/helmet_detection/weights/best.pt"

    if os.path.exists(trained_model_path):
        new_model_path = r"C:\Users\gaurm\OneDrive\Desktop\YoLO\trained_helmet_model.pt"
        model = YOLO(trained_model_path)
        model.export(format="torchscript", path=new_model_path)
        print(f"Model saved successfully at {new_model_path}")
    else:
        print("Error: Model file not found!")

if __name__ == "__main__":
    train_yolo()
