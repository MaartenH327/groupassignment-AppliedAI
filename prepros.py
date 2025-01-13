# Import required libraries
import os
import torch
from torchvision import io, transforms as v2

# Set device to CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define input and output directories
input_folder = "rawimages"  # Folder with raw JPG images
output_folder = "processed_imagesV2"  # Folder to save processed images
os.makedirs(output_folder, exist_ok=True)

# Set transformation pipeline
transform_pipeline = v2.Compose([
    v2.Resize(size=(256, 256)),  # Downscale the image to 256x256 first
    v2.CenterCrop(size=(224, 224)),  # Then randomly crop the image to 224x224
    v2.RandomHorizontalFlip(p=0.5),  # Horizontal flip with probability 0.5
    v2.ConvertImageDtype(torch.float32),  # Convert image to float32
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Process and save images
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(".jpg"):  # Process only .jpg files
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # Load image as a tensor and move it to the GPU (if available)
        img = io.read_image(input_path).to(device)  # Read image and move to device
        
        # If the image is grayscale (single channel), convert it to RGB by repeating the channel
        if img.shape[0] == 1:  # Handle grayscale images
            img = img.expand(3, -1, -1)  # Convert grayscale to RGB (3 channels)

        # Apply transformations
        processed_img = transform_pipeline(img)

        # Denormalize and move back to CPU for saving
        processed_img = processed_img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        processed_img = processed_img.clamp(0, 1) * 255  # Scale back to uint8

        # Move the processed image back to CPU before saving (JPEG saving works on CPU)
        processed_img = processed_img.byte().cpu()  # Convert to uint8 and move to CPU
        io.write_jpeg(processed_img, output_path)

print(f"Processed images saved in {output_folder}")
