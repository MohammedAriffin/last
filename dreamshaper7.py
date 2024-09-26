import torch
import os
from diffusers import LCMScheduler, AutoPipelineForText2Image

# Define the directory where you want to save the images and count
output_dir = "Image generates"
count="count.txt"

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if os.path.exists(count):
    with open(count,"r") as file:
        current_count=int(file.read())
else:
    current_count=0
model_id = "Lykon/dreamshaper-7"
adapter_id = "latent-consistency/lcm-lora-sdv1-5"

pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# load and fuse lcm lora
pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()

# disable guidance_scale by passing 0
n = int(input("Enter how many images to generate: "))
prompt = "girl won the marathon and sharing her medal with her teammates"
for i in range(n):
    image = pipe(prompt=prompt, num_inference_steps=12, guidance_scale=0, width = 1024, height=1024).images[0]
    image.save(os.path.join(output_dir,f"image{current_count}.png"))
    current_count+=1
#576 1024
with open(count, 'w') as file:
    file.write(str(current_count))
print(f"{n} images have been generated and saved in the '{output_dir}' folder.")