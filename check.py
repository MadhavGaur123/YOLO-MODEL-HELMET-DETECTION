import os

image_dir = r"C:\Users\gaurm\Downloads\datasetYOLO\images\train"
label_dir = r"C:\Users\gaurm\Downloads\datasetYOLO\labels\train"

image_files = {f.replace(".jpg", "").replace(".png", "") for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))}
label_files = {f.replace(".txt", "") for f in os.listdir(label_dir) if f.endswith(".txt")}

missing_labels = image_files - label_files
missing_images = label_files - image_files

if missing_labels:
    print("🚨 Missing labels for images:", missing_labels)
if missing_images:
    print("🚨 Missing images for labels:", missing_images)

if not missing_labels and not missing_images:
    print("✅ All images have matching labels!")
