import torch
import os
import cv2
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
import numpy as np
from transformers import AutoProcessor, CLIPModel
# Import functions from functions.py
from functions import (
    file_system,image_read, encode_image, img2edge, extract_clip_features,
    pad_image, load_negative_prompts,save_generated_image
) 
# Define output directory and count file
output_dir = "Image/canny_images"
count_file = "image_count.txt"

# Adjust image size for the target output resolution
target_resolution = (1080, 1920)  # e.g., 1080x1350

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the current image count or initialize it to 0
if os.path.exists(count_file):
    with open(count_file, "r") as file:
        current_count = int(file.read())
else:
    current_count = 0
current_count,output_dir,count_file=file_system()

#image inputs
canny_image_path= "templates/img10.png"

scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-canny-sdxl-1.0", torch_dtype=torch.float16)

pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    adapter=adapter,
    vae=vae,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to("cuda")

# Speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Enable CPU offloading to manage VRAM usage
pipe.enable_model_cpu_offload()

negative_prompts = load_negative_prompts("negative.txt")

# Function for image generation
def generate_image(prompt, num_inference_steps=60, guidance_scale=9):
    try:
        # Generate a random Canny edge map (since there is no initial image)
        #random_noise = torch.randn(1, 3, *target_resolution[::-1]).to("cuda")
        #canny_image = img2edge(random_noise.cpu().numpy().astype(np.uint8).squeeze())
        canny_image = img2edge(image_read(canny_image_path))
        # Fixed random noises for generations with same prompts
        generator = torch.manual_seed(20)

        # Generate image using the pipeline
        generated_image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            #generator=generator,
            negative_prompt=negative_prompts,
            width=target_resolution[0],
            height=target_resolution[1],
            requires_safety_checker=False,
            image=canny_image,
            controlnet_conditioning_scale=0.5,
            controlnet_guidance_start=0.0,
            controlnet_guidance_end=0.35,
            eta=0.4,
        ).images[0]

        return generated_image
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

# User input for generations
turns = int(input("How many generations (each generation will generate 3 images): "))

for i in range(turns):
    # Ask for 3 prompts
    prompts = []
    for j in range(3):
        prompt = input(f"Enter prompt {j + 1} for generation {i + 1}: ")
        prompts.append(prompt)

    # Generate and save an image for each prompt
    for j, prompt in enumerate(prompts):
        # Function call to generate image
        temp = generate_image(prompt=prompt)

        if temp:
            # Save the image with the filename based on the current count
            image_filename = f"image_{current_count}.png"
            temp.save(os.path.join(output_dir, image_filename))

            # Update and save the new count
            current_count += 1
            with open(count_file, "w") as file:
                file.write(str(current_count))

            print(f"Image saved as {image_filename}")
        else:
            print(f"Failed to generate image for prompt: {prompt}")

print(f"{turns * 3} images have been generated and saved in the '{output_dir}' folder.")
