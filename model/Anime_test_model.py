import os
import cv2
import numpy as np
import onnxruntime as ort
from glob import glob

# Define supported image formats
pic_form = ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']


def check_folder(path):
    """Create a folder if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def process_image(img):
    """Resize and normalize the image for model input."""
    h, w = img.shape[:2]

    # Resize dimensions to the nearest multiple of 32
    def to_32s(x):
        return 256 if x < 256 else x - x % 32

    img = cv2.resize(img, (to_32s(w), to_32s(h)))  # Resize the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0  # Normalize
    return img


def load_model(model_file):
    """Load the ONNX model for inference."""
    return ort.InferenceSession(model_file)


def save_image(image, image_path):
    """Save the transformed image after scaling back to [0, 255]."""
    image = (np.squeeze(image) + 1) / 2 * 255  # Scale to [0, 255]
    image = np.clip(image, 0, 255).astype(np.uint8)  # Ensure valid pixel values
    cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Save image


def transform_images(model_path, image_folder, output_folder):
    """Transform images in the specified folder using the ONNX model."""
    ort_session = load_model(model_path)  # Load the ONNX model
    check_folder(output_folder)  # Create output folder if it doesn't exist

    # Get list of image files in the folder
    image_files = glob(f'{image_folder}/*.*')
    image_files = [f for f in image_files if os.path.splitext(f)[-1] in pic_form]

    # Process each image file
    for i, image_file in enumerate(image_files):
        input_image = cv2.imread(image_file).astype(np.float32)  # Read image and convert to float32
        if input_image is None:
            print(f"Warning: Could not read image {image_file}. Skipping...")
            continue

        processed_image = process_image(input_image)  # Preprocess the image
        processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

        # Run the model on the input image
        output_image = ort_session.run(None, {ort_session.get_inputs()[0].name: processed_image})[0]

        # Save the output image
        output_image_path = os.path.join(output_folder, f'anime_{os.path.basename(image_file)}')
        save_image(output_image, output_image_path)

        print(f'Processed image: {i + 1}/{len(image_files)}, saved anime-style image: {output_image_path}')


# Update the model path and output folder path
model_path = "model/AnimeGANv2/pb_and_onnx_model/Shinkai_53.onnx"  # Path to your model file
image_folder = "saved_images"  # Path to your folder containing images
output_folder = "anime_output"  # Path to your folder for saving anime-styled images

# Start the transformation process
transform_images(model_path, image_folder, output_folder)
