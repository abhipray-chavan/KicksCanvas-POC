import os 
import cv2
import numpy as np 
from PIL import Image 

# --- Configuration ---
source_folder = "images"
output_folder = "control_maps"
num_images = 31

# creating the output folder if it does not exist 
os.makedirs(output_folder, exist_ok=True) 

print("Processing images...")
for i in range(1, num_images + 1):
    image_path = os.path.join(source_folder, f"{i}.jpg")
    
    if not os.path.exists(image_path):
        print(f"Warning: Source image {image_path} not found. Skipping.")
        continue
    
    output_path = os.path.join(output_folder, f"{i}_canny.png")
    
    # reading the image using opencv 
    image = cv2.imread(image_path)
    # converting the image to grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # applying canny edge detection 
    edges = cv2.Canny(gray, 100, 200)
    # converting the image to PIL format 
    pil_image = Image.fromarray(edges)
    # saving the image 
    pil_image.save(output_path)
    print(f"Processed image {i} -> {output_path}")

print("Done!")