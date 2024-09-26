import torch
import os
import cv2
from PIL import Image , ImageOps
from diffusers import ControlNetModel, UniPCMultistepScheduler , StableDiffusionControlNetImg2ImgPipeline , AutoencoderKL ,T2IAdapter
import numpy as np
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel

# Define the directory where you want to save the images and count
output_dir = "Images generated"
count_file = "count.txt"

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
seed_image_path = "templates/img4.png"  # Replace with your image path
if not os.path.exists(seed_image_path):
    raise ValueError(f"Seed image file not found: {seed_image_path}")

# Load the SDXL v1.5 model and Canny ControlNet
model_id = "runwayml/stable-diffusion-v1-5"
control_id="lllyasviel/sd-controlnet-canny"
adapter = T2IAdapter.from_pretrained("TencentARC/t2iadapter_canny_sd15v2", torch_dtype=torch.float16)

# Set the negative prompt
negative = "low quality, bad quality, sketches, ugly, deformed, disfigured, poor details, bad anatomy, multiple people, crowd"

# Initialize the Stable Diffusion pipeline with Canny ControlNet
controlnet = ControlNetModel.from_pretrained(control_id, torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    adapter=adapter,
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

# Resize seed image to the target resolution
def resize_image(image, target_resolution):
    """Resizes an image to the specified target resolution.

    Args:
        image: The input image as a PIL Image or NumPy array.
        target_resolution: A tuple (width, height) specifying the desired resolution.

    Returns:
        The resized image as a PIL Image.
    """

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    width, height = target_resolution
    resized_image = image.resize((width, height), Image.BICUBIC)  # Use BICUBIC interpolation for better quality
    return resized_image

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

    target_height, target_width = target_resolution[1] // 8, target_resolution[0] // 8
    latents_resized = F.interpolate(latents, size=(target_height, target_width), mode='bilinear', align_corners=False)
    return latents_resized



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

#function for image generation
def generate_image(prompt, seed_image=None, num_inference_steps=75, guidance_scale=15):
    try:
        # Prepare seed image using OpenCV and convert to PIL Image if seed image is provided
        if seed_image:
            initial_image = image_read(seed_image)
            # Resize the seed image
            initial_image = resize_image(initial_image, target_resolution)

            #Encode the seed image
            encoded_seed = encode_image(initial_image, vae)
            # Encode the resized seed image
            print(f"Encoded seed shape: {encoded_seed.shape}")
            

            # convert image to edges / dots for controlnet mapping
            canny_image = img2edge(initial_image)
            # Resize the Canny image to match the target resolution
            canny_image = resize_image(canny_image, target_resolution)
            print(f"Canny image shape: {np.array(canny_image).shape}")
            

            clip_features= extract_clip_features(seed_image)
            print(f"CLIP features shape: {clip_features.shape}")

        else:
            initial_image = None
            canny_image = None

        if initial_image is None or canny_image is None or clip_features is None:
            raise ValueError("One of the images or features is None, cannot proceed.")

        # Fixed random noises for generations with same prompts
        generator = torch.manual_seed(150)

        # Generate image using the pipeline
        generated_image = pipe(
            prompt=prompt,
            image=initial_image,
            #latent=encoded_seed,
            #generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative,
            width=target_resolution[0],
            height=target_resolution[1],
            requires_safety_checker= False,
            safety_checker=None,
            control_image=canny_image,
            controlnet_conditioning_scale=0.1,
            controlnet_guidance_start=0.1,
            controlnet_guidance_end=0.3,
            adapter_conditioning_scale=0.4,
            eta=0.1,
            # You might need to add a custom method to pipe to use these features
            #clip_features=clip_features,
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