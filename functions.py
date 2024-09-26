# functions.py
import torch
import cv2,os
import numpy as np
from PIL import Image, ImageOps
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

def image_read(seed_image):
    """Reads an image using OpenCV and converts it to a PIL Image."""
    initial_image = cv2.imread(seed_image)
    if initial_image is None:
        raise ValueError(f"Could not load seed image: {seed_image}")
    initial_image = cv2.cvtColor(initial_image, cv2.COLOR_BGR2RGB)
    initial_image = Image.fromarray(initial_image)
    return initial_image

def encode_image(image, vae, target_resolution):
    """Encodes an image using the Variational Autoencoder (VAE)."""
    # Convert PIL Image to tensor
    image_tensor = processor(images=image, return_tensors="pt").pixel_values.to("cuda")
    
    # Encode the image using VAE
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    target_height, target_width = target_resolution[1] // 8, target_resolution[0] // 8
    latents_resized = F.interpolate(latents, size=(target_height, target_width), mode='bilinear', align_corners=False)
    return latents_resized

def img2edge(image):
    """Converts an image to Canny edge maps."""
    np_image = np.array(image)
    np_image = cv2.Canny(np_image, 100, 200)
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    canny_image = Image.fromarray(np_image)
    return canny_image

def extract_clip_features(image_path):
    """Extracts CLIP features from an image."""
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs).to("cuda")  # Move to GPU if necessary
    return image_features

def pad_image(image, target_resolution):
    """Pads an image to the target resolution while maintaining the original aspect ratio."""
    width, height = image.size
    target_width, target_height = target_resolution
    
    # Calculate padding to center the image
    delta_width = target_width - width
    delta_height = target_height - height
    padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
    
    # Pad the image
    padded_image = ImageOps.expand(image, padding, (0, 0, 0))
    return padded_image

def load_negative_prompts(file_path):
    """Loads and joins negative prompts from a text file as a single string."""
    with open(file_path, "r") as f:
        negative_prompts = f.read().splitlines()
    # Join the list into a single string, separated by commas
    return ', '.join(negative_prompts)

def file_system():
    
    # Define output directory and count file
    output_dir = "Image/canny_images"
    count_file = "image_count.txt"
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the current image count or initialize it to 0
    if os.path.exists(count_file):
        with open(count_file, "r") as file:
            current_count = int(file.read())
    else:
        current_count = 0
    
    return current_count,output_dir,count_file

def save_generated_image(image, output_dir, count_file, current_count):
    """
    Saves the generated image to the output directory and updates the image count.
    
    Args:
        image (PIL.Image): The generated image to save.
        output_dir (str): The directory where images should be saved.
        count_file (str): The file that tracks the number of saved images.
        current_count (int): The current image count.
    
    Returns:
        tuple: A tuple containing the new seed image path and the updated image count.
    """
    # Save the image with the filename based on the current count
    image_filename = f"image_{current_count}.png"
    image_path = os.path.join(output_dir, image_filename)
    image.save(image_path)

    # Update the image count
    current_count += 1

    # Save the new count to the count file
    with open(count_file, "w") as file:
        file.write(str(current_count))

    return image_path, current_count

def resize_and_center_crop(image, target_resolution):
    """
    Resize and center crop an image to the target resolution.

    Args:
        image (PIL.Image): The input image.
        target_resolution (tuple): The target resolution (width, height).
    
    Returns:
        PIL.Image: The resized and cropped image.
    """
    # Resize the image while maintaining the aspect ratio
    image.thumbnail(target_resolution, Image.ANTIALIAS)

    # Get dimensions of the resized image
    width, height = image.size

    # Calculate the cropping box to center the image
    left = (width - target_resolution[0]) // 2
    top = (height - target_resolution[1]) // 2
    right = (width + target_resolution[0]) // 2
    bottom = (height + target_resolution[1]) // 2

    # Crop the image to the target resolution
    image = image.crop((left, top, right, bottom))
    
    return image

def get_target_resolution(input_image, max_width=1080, max_height=1350):
    """
    Calculate target resolution based on the aspect ratio of the input image.
    
    Args:
        input_image (PIL.Image): The input image to base the resolution on.
        max_width (int): The maximum width of the output image.
        max_height (int): The maximum height of the output image.
    
    Returns:
        tuple: A tuple containing the width and height of the target resolution.
    """
    # Get input image dimensions
    input_width, input_height = input_image.size

    # Calculate aspect ratio
    aspect_ratio = input_width / input_height

    # Determine target dimensions while maintaining the aspect ratio
    if input_width > input_height:
        target_width = max_width
        target_height = int(max_width / aspect_ratio)
    else:
        target_height = max_height
        target_width = int(max_height * aspect_ratio)

    # Ensure the dimensions do not exceed the max limits
    target_width = min(target_width, max_width)
    target_height = min(target_height, max_height)

    return target_width, target_height