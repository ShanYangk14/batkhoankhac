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

def load_arcane_model(model_id="nitrosocke/Arcane-Diffusion"):
    """Load the arcane diffusion model."""
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    return pipe

def process_arcane_image(pipe, img_path, prompt=None):
    """Generate a subtle arcane-style enhancement for a selfie image."""
    default_prompt = (
         "arcane style, rich colors, high contrast, stylized lighting, dramatic shadows"
    )
    # Use the user-provided prompt if available, otherwise use the default prompt
    prompt = prompt or default_prompt
    # Load and preprocess image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Could not read image {img_path}. Skipping...")
        return None

    # Convert image to RGB and resize to 512x512 (default size for diffusion models)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))

    # Debugging line to check input image format (PIL format)
    pil_image = Image.fromarray(image)
    print(f"Original image type (PIL): {type(pil_image)}")

    # Apply the Img2Img transformation with the prompt
    with torch.no_grad():
        generated_image_pil = pipe(
            prompt=prompt,
            image=pil_image,  # PIL.Image format
            strength=0.42,
            guidance_scale = 7.5
        ).images[0]
        print("Generated image from PIL format.")

    # Convert result to numpy array and return the result
    arcane_img = np.array(generated_image_pil)
    return arcane_img

def transform_images_arcane(model_dir, image_folder, output_folder, prompt=None):
    """Process each image in the folder using the arcane model."""
    arcane_pipe = load_arcane_model(model_dir)  # Load the Cyberpunk model
    check_folder(output_folder)  # Ensure the output folder exists

    # Get list of images in the folder
    image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[-1] in pic_formats]

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        # Generate arcane-style image
        arcane_img = process_arcane_image(arcane_pipe, img_path, prompt)

        # Check if the image was processed successfully
        if arcane_img is not None:
            # Save output image
            output_path = os.path.join(output_folder, f"arcane_{img_file}")
            cv2.imwrite(output_path, cv2.cvtColor(arcane_img, cv2.COLOR_RGB2BGR))
            print(f"Processed and saved: {output_path}")

# Define model path and directories
model_dir = "nitrosocke/Arcane-Diffusion"
image_folder = r"C:\Users\admin\PycharmProjects\batkhoankhac\uploads"  # Folder containing the input images
output_folder = r"C:\Users\admin\PycharmProjects\batkhoankhac\arcane_output"  # Folder to save the generated cyberpunk-style images

# Run the image transformation
transform_images_arcane(model_dir, image_folder, output_folder)
