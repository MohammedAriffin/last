import torch
import os
from huggingface_hub import hf_hub_download
from diffusers import ControlNetModel , StableDiffusionControlNetImg2ImgPipeline , AutoencoderKL , LMSDiscreteScheduler
import numpy as np
from transformers import AutoProcessor, CLIPModel
# Import functions from functions.py
from functions import (
    file_system,image_read, encode_image, img2edge, extract_clip_features,
    pad_image, load_negative_prompts,save_generated_image
) 

# Adjust image size for the target output resolution
target_resolution = (736, 1223)  # e.g., 1080x1350
current_count,output_dir,count_file=file_system()
# Load your seed image using OpenCV
seed_image_path = "templates/img14.png"  # Replace with your image path

if not os.path.exists(seed_image_path):
    raise ValueError(f"Seed image file not found: {seed_image_path}")

# Load the SDXL v1.5 model and Canny ControlNet
model_id = "runwayml/stable-diffusion-v1-5"
canny_control_id="lllyasviel/sd-controlnet-canny"
openpose_control_id = "lllyasviel/sd-controlnet-openpose"

# Initialize the Stable Diffusion pipeline with Canny ControlNet
controlnet = ControlNetModel.from_pretrained(canny_control_id, torch_dtype=torch.float16).to("cuda")
openpose_controlnet = ControlNetModel.from_pretrained(openpose_control_id, torch_dtype=torch.float16).to("cuda")

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

# Enable CPU offloading to manage VRAM usage
pipe.enable_model_cpu_offload()

negative_prompts = load_negative_prompts("negative.txt")

#function for image generation
def generate_image(prompt, seed_image=None, num_inference_steps=60, guidance_scale=14):
    try:
        # Prepare seed image using OpenCV and convert to PIL Image if seed image is provided
        if seed_image:
            initial_image = image_read(seed_image)

            #Encode the seed image
            #encoded_seed = encode_image(initial_image, vae)
            #print(f"Encoded seed shape: {encoded_seed.shape}")
            
            #convert image to edges / dots for controlnet mapping
            canny_image = img2edge(image_read(seed_image))
            print(f"Canny image shape: {np.array(canny_image).shape}")

        else:
            initial_image = None
            canny_image = None

        if initial_image is None or canny_image is None :
            raise print("One of the images or features is None, cannot proceed.")

        # Fixed random noises for generations with same prompts
        #generator = torch.manual_seed(200000)
        # Generate image using the pipeline
        generated_image = pipe(
            prompt=prompt,
            image=initial_image,
            #generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompts,
            width=target_resolution[0],
            height=target_resolution[1],
            requires_safety_checker= False,
            control_image=canny_image,
            controlnet_conditioning_scale=0.6,
            controlnet_guidance_start=0.0,
            controlnet_guidance_end=0.3,
            eta=0.3,
            #strength=0.7,
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
