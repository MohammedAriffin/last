import torch
import os
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
import numpy as np
from transformers import AutoProcessor, CLIPModel
# Import functions from functions.py
from functions import (
    file_system, image_read, encode_image, img2edge, extract_clip_features,
    pad_image, load_negative_prompts, save_generated_image
)

# Adjust image size for the target output resolution
target_resolution = (608 , 760)  # e.g., 1080x1350
current_count, output_dir, count_file = file_system()

# Load your seed image using OpenCV and  Replace with your image path
seed_image_path = "templates/img4.png"  
if not os.path.exists(seed_image_path):
    raise ValueError(f"Seed image file not found: {seed_image_path}")

# Load the SDXL v1.5 model
model_id = "runwayml/stable-diffusion-v1-5"

# Initialize the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to("cuda")

# Load the IP-Adapter
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")
pipe.set_ip_adapter_scale(0.6) #weight of Ip adapter

# Speed up the diffusion process with a faster scheduler and memory optimization
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Enable CPU offloading to manage VRAM usage
pipe.enable_model_cpu_offload()

#load negative prompts intialised from a text file
negative_prompts = load_negative_prompts("negative.txt")

# Function for image generation
def generate_image(prompt, seed_image=None, num_inference_steps=100, guidance_scale=5):
    try:
        # Prepare seed image using OpenCV and convert to PIL Image if seed image is provided
        initial_image = image_read(seed_image) if seed_image else None
        adapter_image = image_read(seed_image) if seed_image else None

        if initial_image is None or adapter_image is None:
            raise ValueError("Initial image or IP-Adapter image is None, cannot proceed.")

        # Generate image using the pipeline
        generated_image = pipe(
            prompt=prompt,
            image=initial_image,  # The base image for the generation
            ip_adapter_image=adapter_image,  # The image for facial features
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompts,
            width=target_resolution[0],
            height=target_resolution[1],
            requires_safety_checker=False,
            eta=0.3, #scheduler beta values
            strength=0.3, #higher the value , loser the priority of image details
        ).images[0]

        return generated_image
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

# User input for generations
turns = int(input("How many generations: "))

for i in range(turns):
    prompt = input(f"Enter prompt for generation {i + 1}: ")

    # Function call to generate image
    temp = generate_image(prompt=prompt, seed_image=seed_image_path)

    if temp:
        # Use the new function to save the generated image and update count
        seed, current_count = save_generated_image(temp, output_dir, count_file, current_count)
        print(f"Image saved as {os.path.basename(seed)}")
    else:
        print(f"Failed to generate image for prompt: {prompt}")
