import cv2
import torch
from ultralytics import YOLO


model_path = r"C:\Users\gaurm\OneDrive\Desktop\YoLO\hemletYoloV8_100epochs.pt"
model = YOLO(model_path)


video_path = r"C:\Users\gaurm\Downloads\archive\videos_infer\1.mp4"  
cap = cv2.VideoCapture(video_path)


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
out = cv2.VideoWriter('output_helmet_detection.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    helmet_detected = False
    results = model(frame_rgb)
    for result in results:
        boxes = result.boxes  
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class ID
            label = model.model.names[cls]  # Get class name
            if label.lower() == "helmet":  
                helmet_detected = True
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display label
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    if helmet_detected:
        cv2.putText(frame, "Helmet detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Warning: No helmet detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write processed frame to output video
    out.write(frame)

    # Show the frame (optional)
    cv2.imshow('Helmet Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
