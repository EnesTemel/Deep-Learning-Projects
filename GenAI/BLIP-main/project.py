from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import logging
from pathlib import Path
from diffusers import StableDiffusionPipeline
from fastcore.all import concat
from diffusers import StableDiffusionImg2ImgPipeline
import os

SDV5_MODEL_PATH = r'C:\Users\enesm\stable-diffusion-v1-5'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path= r"C:\Users\enesm\OneDrive\Masaüstü\P432.671__S_0__1b744e9a20d14e08b18b40cfb1f40834.jpg"
raw_image = Image.open(path).convert('RGB')

def load_demo_image(image_size,device,image):
    w,h = raw_image.size    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image


from models.blip import blip_decoder

image_size = 720
image = load_demo_image(image_size=image_size, device=device,image=raw_image)

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

with torch.no_grad():
    # beam search
    caption = model.generate(image, sample=False, num_beams=1, max_length=20, min_length=5) 
    # nucleus sampling
    # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
    print('caption:'+caption[0])
model.to('cpu')

torch.cuda.empty_cache()

def image_grid(imgs,rows,cols):
    w,h = imgs[0].size
    grid = Image.new('RGB', size = (cols*w , rows*h))
    for i, img in enumerate(imgs): grid.paste(img,box=(i%cols*w, i//cols*h))
    return grid


pipe = StableDiffusionImg2ImgPipeline.from_pretrained(SDV5_MODEL_PATH, revision="fp16", torch_dtype=torch.float16).to("cuda")
prompt = caption[0]
images = pipe(prompt=prompt, num_images_per_prompt=1, image=raw_image, strength=0.9, num_inference_steps=50).images
image_grid(images, rows=1, cols=3)