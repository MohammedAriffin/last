import torch
import os
from huggingface_hub import hf_hub_download
from diffusers import ControlNetModel, DDIMScheduler , StableDiffusionControlNetImg2ImgPipeline
import numpy as np
# Import functions from functions.py
from functions import (
    file_system,image_read, encode_image, img2edge, extract_clip_features,
    pad_image, load_negative_prompts,save_generated_image
) 

# Adjust image size for the target output resolution
target_resolution = (525, 933)  # e.g., 1080x1350
current_count,output_dir,count_file=file_system()

# Load your seed image using OpenCV
seed_image_path = "templates/img11.png"  # Replace with your image path
canny_image_path= "templates/img11.png"
adapter_image_path = "templates/face2.png"

if not os.path.exists(seed_image_path):
    raise ValueError(f"Seed image file not found: {seed_image_path}")

# Load the SDXL v1.5 model and Canny ControlNet
model_id = "runwayml/stable-diffusion-v1-5"
canny_control_id="lllyasviel/sd-controlnet-canny"
openpose_control_id = "lllyasviel/sd-controlnet-openpose"

# Initialize the Stable Diffusion pipeline with Canny ControlNet
controlnet = ControlNetModel.from_pretrained(canny_control_id, torch_dtype=torch.float16).to("cuda")
#openpose_controlnet = ControlNetModel.from_pretrained(openpose_control_id, torch_dtype=torch.float16).to("cuda")

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

# Load the IP-Adapter
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")
pipe.set_ip_adapter_scale(0.6)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Enable CPU offloading to manage VRAM usage
pipe.enable_model_cpu_offload()

negative_prompts = load_negative_prompts("negative.txt")

#function for image generation
def generate_image(prompt, seed_image=None , num_inference_steps=100, guidance_scale=9):
    try:
        # Prepare seed image using OpenCV and convert to PIL Image if seed image is provided
        if seed_image:
            initial_image = image_read(seed_image) if seed_image else None
            canny_image = img2edge(image_read(seed_image))
            adapter_image = image_read(seed_image)
            print(f"Canny image shape: {np.array(canny_image).shape if canny_image is not None else 'None'}")
            print(f"IP-Adapter image shape: {np.array(adapter_image).shape if adapter_image is not None else 'None'}")
        else:
            initial_image = None
            canny_image = None

        if initial_image is None:
            raise print("One of the images or features is None, cannot proceed.")

        # Fixed random noises for generations with same prompts
        #generator = torch.manual_seed(200000)
        # Generate image using the pipeline
        generated_image = pipe(
            prompt=prompt,
            image=initial_image,
            #generator=generator,
            ip_adapter_image=adapter_image, # The image for facial features (IP Adapter)
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompts,
            width=target_resolution[0],
            height=target_resolution[1],
            requires_safety_checker= False,
            control_image=canny_image, # ControlNet Canny image
            controlnet_conditioning_scale=0.6,
            controlnet_guidance_start=0.0,
            controlnet_guidance_end=0.3,
            eta=0.3,
            strength=0.6, # Strength parameter for image guidance
        ).images[0]
        
        return generated_image
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

# User input for generations
turns = int(input("How many generations: "))
seed = seed_image_path
for i in range(turns):
    prompt = input(f"Enter prompt for generation {i + 1}: ")

    # Function call to generate image
    temp = generate_image(prompt=prompt, seed_image=seed)

    if temp:
        # Use the new function to save the generated image and update count
        seed, current_count = save_generated_image(temp, output_dir, count_file, current_count)
        print(f"Image saved as {os.path.basename(seed)}")
    else:
        print(f"Failed to generate image for prompt: {prompt}")
