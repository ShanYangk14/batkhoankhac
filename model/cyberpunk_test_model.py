import os
import cv2
import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from huggingface_hub import login

token = "HUGGINGFACE_TOKEN"
login(token)

# Supported image formats
pic_formats = ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']

def check_folder(path):
    """Create a folder if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def load_cyberpunk_model(model_id="DGSpitzer/Cyberpunk-Anime-Diffusion"):
    """Load the Cyberpunk diffusion model."""
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    return pipe

def process_cyberpunk_image(pipe, img_path, prompt=None):
    """Generate a cyberpunk-style enhancement using a custom or default prompt."""
    # Define the default prompt
    default_prompt = (
        "dgs illustration style, Anime fine details portrait in front of modern night city landscape on the background, anime masterpiece, 8k, sharp high quality anime"
    )
    # Use the user-provided prompt if available, otherwise use the default prompt
    prompt = prompt or default_prompt

    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Could not read image {img_path}. Skipping...")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))

    pil_image = Image.fromarray(image)

    with torch.no_grad():
        generated_image_pil = pipe(
            prompt=prompt,
            image=pil_image,
            strength=0.35,
            guidance_scale=6.8
        ).images[0]
        print("Generated image from PIL format.")

    cyberpunk_img = np.array(generated_image_pil)
    return cyberpunk_img


def transform_images_cyberpunk(model_dir, image_folder, output_folder, prompt=None):
    """Process each image in the folder using the cyberpunk model."""
    cyberpunk_pipe = load_cyberpunk_model(model_dir)  # Load the Cyberpunk model
    check_folder(output_folder)  # Ensure the output folder exists

    # Get list of images in the folder
    image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[-1] in pic_formats]

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        # Generate cyberpunk-style image with the provided prompt
        cyberpunk_img = process_cyberpunk_image(cyberpunk_pipe, img_path, prompt)

        # Check if the image was processed successfully
        if cyberpunk_img is not None:
            # Save output image
            output_path = os.path.join(output_folder, f"cyberpunk_{img_file}")
            cv2.imwrite(output_path, cv2.cvtColor(cyberpunk_img, cv2.COLOR_RGB2BGR))
            print(f"Processed and saved: {output_path}")

# Define model path and directories
model_dir = "DGSpitzer/Cyberpunk-Anime-Diffusion"
image_folder = r"C:\Users\admin\PycharmProjects\batkhoankhac\uploads"  # Folder containing the input images
output_folder = r"C:\Users\admin\PycharmProjects\batkhoankhac\cyberpunk_output"  # Folder to save the generated cyberpunk-style images

# Run the image transformation
transform_images_cyberpunk(model_dir, image_folder, output_folder)
