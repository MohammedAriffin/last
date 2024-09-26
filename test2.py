import torch
import os
import cv2
from PIL import Image , ImageOps
from diffusers import ControlNetModel, UniPCMultistepScheduler , StableDiffusionControlNetImg2ImgPipeline , AutoencoderKL
import numpy as np
from transformers import AutoProcessor, CLIPModel

# Define output directory and count file
output_dir = "Image/canny_images"
count_file = "image_count.txt"

# Adjust image size for the target output resolution
target_resolution = (1080, 1350)  # e.g., 1080x1350

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
seed_image_path = "templates/img7.png"  # Replace with your image path
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
    enable_pag=True,
    pag_applied_layers=["mid"],
).to("cuda")

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
# Load pre-trained VAE
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cuda")

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Enable CPU offloading to manage VRAM usage
pipe.enable_model_cpu_offload()

#read image
def image_read(seed_image):
    initial_image = cv2.imread(seed_image)
    if initial_image is None:
        raise ValueError(f"Could not load seed image: {seed_image}")
    initial_image = cv2.cvtColor(initial_image, cv2.COLOR_BGR2RGB)
    initial_image = Image.fromarray(initial_image)
    return initial_image

def encode_image(image, vae):
    # Convert PIL Image to tensor
    image_tensor = processor(images=image, return_tensors="pt").pixel_values.to("cuda")
    
    # Encode the image using VAE
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    return latents



#convert image to canny maps
def img2edge(image):
    np_image = np.array(image)
    np_image = cv2.Canny(np_image, 100, 200)
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    canny_image = Image.fromarray(np_image)
    return canny_image

# Function to extract CLIP features
def extract_clip_features(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs).to("cuda")  # Move to GPU if necessary
    return image_features

# Load and join negative prompts as a single string
def load_negative_prompts(file_path):
    with open(file_path, "r") as f:
        negative_prompts = f.read().splitlines()
    # Join the list into a single string, separated by commas
    return ', '.join(negative_prompts)

negative_prompts = load_negative_prompts("negative.txt")

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

#function for image generation
def generate_image(prompt, seed_image=None, num_inference_steps=50, guidance_scale=6):
    try:
        # Prepare seed image using OpenCV and convert to PIL Image if seed image is provided
        if seed_image:
            initial_image = image_read(seed_image)
            # Resize the seed image
            # Pad the image to the target resolution
            #initial_image = pad_image(initial_image, target_resolution)

            #Encode the seed image
            encoded_seed = encode_image(initial_image, vae)
            print(f"Encoded seed shape: {encoded_seed.shape}")
            

            #convert image to edges / dots for controlnet mapping
            canny_image = img2edge(initial_image)
            # Resize the Canny image to match the target resolution
            #canny_image = pad_image(canny_image, target_resolution)
            print(f"Canny image shape: {np.array(canny_image).shape}")
            

            clip_features= extract_clip_features(seed_image)
            print(f"CLIP features shape: {clip_features.shape}")

        else:
            initial_image = None
            canny_image = None

        if initial_image is None or canny_image is None or clip_features is None:
            raise print("One of the images or features is None, cannot proceed.")

        generator=torch.manual_seed(2)
        # Generate image in latent space
        with torch.no_grad():
            generated_latents = pipe(
                prompt=prompt,
                latent_image=encoded_seed,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompts,
                control_image=canny_image,
                controlnet_conditioning_scale=0.4,
                controlnet_guidance_start=0.0,
                controlnet_guidance_end=0.2,
                eta=0.1,
            ).latent_images[0]

        # Decode latents to image
        generated_image = vae.decode(generated_latents).sample()
        generated_image = (generated_image / 2 + 0.5).clamp(0, 1)
        generated_image = generated_image.permute(1, 2, 0).cpu().numpy()
        generated_image = Image.fromarray((generated_image * 255).astype(np.uint8))
        return generated_image
    
    except Exception as e:
        print(f"Error generating image: {e}")
        return None



# User input for generations
turns = int(input("How many generations: "))
seed = seed_image_path
for i in range(turns):
    prompt = input(f"Enter prompt for generation {i + 1}: ")

    #function call to generate imae
    temp = generate_image(prompt=prompt, seed_image=seed)
    
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

print(f"{turns} images have been generated and saved in the '{output_dir}' folder.")