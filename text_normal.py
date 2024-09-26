import torch
import cv2
from PIL import Image
from diffusers import DiffusionPipeline
import os
#/runwaymlstable-diffusion-v1-5
model = ["runwayml/stable-diffusion-v1-5","stabilityai/stable-diffusion-xl-base-1.0"]
n=int(input("Enter which model \n 1. runway\n 2.sdxl_base "))
# Initialize the Stable Diffusion pipeline
pipe = DiffusionPipeline.from_pretrained(
    model[n-1], 
    use_safetensors=True, 
    variant="fp16"
).to("cuda")

# Enable CPU offloading to manage VRAM usage
pipe.enable_model_cpu_offload()

negative = "low quality, bad quality, sketches, ugly, deformed, disfigured, poor details, bad anatomy,multiple people, crowd"

def generate_image(prompt, seed_image=None, num_inference_steps=50, guidance_scale=3, width=1080, height=1920):
    try:
        # Prepare seed image using OpenCV and convert to PIL Image if seed image is provided
        if seed_image:
            initial_image = cv2.imread(seed_image)
            if initial_image is None:
                raise ValueError(f"Could not load seed image: {seed_image}")
            initial_image = cv2.cvtColor(initial_image, cv2.COLOR_BGR2RGB)
            initial_image = Image.fromarray(initial_image)
        else:
            initial_image = None
        
        # Generate image using the pipeline
        generated_image = pipe(
            prompt=prompt,
            image=initial_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative,
            width=width,
            height=height
        ).images[0]
        
        return generated_image
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

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

# Load your seed image using OpenCV
seed_image_path = "templates/img4.png"  # Replace with your image path
if not os.path.exists(seed_image_path):
    raise ValueError(f"Seed image file not found: {seed_image_path}")

# User input for generations
turns = int(input("How many generations: "))
seed = seed_image_path
for i in range(turns):
    prompt = input(f"Enter prompt for generation {i + 1}: ")
    temp = generate_image(prompt=prompt, seed_image=seed, width=576, height=1024)
    
    if temp:
        # Save the image with the filename based on the current count
        image_filename = f"image_{current_count}.png"
        temp.save(os.path.join(output_dir, image_filename))

        # Update the seed for the next generation
        seed = os.path.join(output_dir, image_filename)
        
        # Update and save the new count
        current_count += 1
        with open(count_file, "w") as file:
            file.write(str(current_count))
        
        print(f"Image saved as {image_filename}")
    else:
        print(f"Failed to generate image for prompt: {prompt}")
