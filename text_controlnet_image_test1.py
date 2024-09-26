import torch
import os
import cv2
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetImg2ImgPipeline, AutoencoderKL, T2IAdapter
import numpy as np
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel

# Setup paths and parameters
output_dir = "Image/canny_images"
count_file = "image_count.txt"
target_resolution = (1080, 1350)
seed_image_path = "templates/img4.png"

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read or initialize image count
if os.path.exists(count_file):
    with open(count_file, "r") as file:
        current_count = int(file.read())
else:
    current_count = 0

# Load models
model_id = "runwayml/stable-diffusion-v1-5"
control_id = "lllyasviel/sd-controlnet-canny"
adapter = T2IAdapter.from_pretrained("TencentARC/t2iadapter_canny_sd15v2", torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained(control_id, torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    adapter=adapter,
).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# Load VAE
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cuda")

# Resize and prepare image
def resize_image(image, target_resolution):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    width, height = target_resolution
    return image.resize((width, height), Image.BICUBIC)

# Encode seed image using VAE
def encode_image(image, vae):
    image_tensor = processor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    target_height, target_width = target_resolution[1] // 8, target_resolution[0] // 8
    return F.interpolate(latents, size=(target_height, target_width), mode='bilinear', align_corners=False)

# Convert image to Canny edges
def img2edge(image):
    np_image = np.array(image)
    np_image = cv2.Canny(np_image, 100, 200)
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    return Image.fromarray(np_image)

# Generate image
def generate_image(prompt, seed_image=None, num_inference_steps=75, guidance_scale=15):
    try:
        if seed_image:
            initial_image = resize_image(image_read(seed_image), target_resolution)
            encoded_seed = encode_image(initial_image, vae)
            canny_image = img2edge(initial_image)
            canny_image = resize_image(canny_image, target_resolution)
        else:
            initial_image, canny_image = None, None

        if initial_image is None or canny_image is None:
            raise ValueError("Initial or Canny image is None.")

        generator = torch.manual_seed(150)
        generated_image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=target_resolution[0],
            height=target_resolution[1],
            control_image=canny_image,
        ).images[0]
        
        return generated_image

    except Exception as e:
        print(f"Error generating image: {e}")
        return None

# Generate images based on user input
turns = int(input("How many generations: "))
seed = seed_image_path

for i in range(turns):
    prompt = input(f"Enter prompt for generation {i + 1}: ")
    temp = generate_image(prompt=prompt, seed_image=seed)
    
    if temp:
        image_filename = f"image_{current_count}.png"
        temp.save(os.path.join(output_dir, image_filename))
        seed = os.path.join(output_dir, image_filename)
        current_count += 1
        with open(count_file, "w") as file:
            file.write(str(current_count))
        print(f"Image saved as {image_filename}")
    else:
        print(f"Failed to generate image for prompt: {prompt}")

print(f"{turns} images have been generated and saved in the '{output_dir}' folder.")
