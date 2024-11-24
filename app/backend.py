import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import sys
import time
import cv2
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from model.Anime_test_model import transform_images, load_model, check_folder
from model.cyberpunk_test_model import transform_images_cyberpunk, load_cyberpunk_model, check_folder, process_cyberpunk_image
from model.arcane_test_model import transform_images_arcane, load_arcane_model, check_folder, process_arcane_image
from huggingface_hub import login

app = Flask(__name__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration for upload folders
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['SAVED_IMAGES_FOLDER'] = os.path.join(os.getcwd(), 'saved_images')
app.config['ANIME_OUTPUT_FOLDER'] = os.path.join(os.getcwd(), 'anime_output')
app.config['CYBERPUNK_OUTPUT_FOLDER'] = os.path.join(os.getcwd(), 'cyberpunk_output')
app.config['ARCANE_OUTPUT_FOLDER'] = os.path.join(os.getcwd(), 'arcane_output')

UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
SAVED_IMAGES_FOLDER = app.config['SAVED_IMAGES_FOLDER']
ANIME_OUTPUT_FOLDER = app.config['ANIME_OUTPUT_FOLDER']
CYBERPUNK_OUTPUT_FOLDER = app.config['CYBERPUNK_OUTPUT_FOLDER']
ARCANE_OUTPUT_FOLDER = app.config['ARCANE_OUTPUT_FOLDER']

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(SAVED_IMAGES_FOLDER):
    os.makedirs(SAVED_IMAGES_FOLDER)
if not os.path.exists(CYBERPUNK_OUTPUT_FOLDER):
    os.makedirs(CYBERPUNK_OUTPUT_FOLDER)
if not os.path.exists(ARCANE_OUTPUT_FOLDER):
    os.makedirs(ARCANE_OUTPUT_FOLDER)

# Directory for saved images
SAVED_IMAGES_FOLDER = os.path.join(os.getcwd(), 'saved_images')
if not os.path.exists(SAVED_IMAGES_FOLDER):
    os.makedirs(SAVED_IMAGES_FOLDER)

for folder in [app.config['UPLOAD_FOLDER'], app.config['SAVED_IMAGES_FOLDER'], app.config['ANIME_OUTPUT_FOLDER'], app.config['CYBERPUNK_OUTPUT_FOLDER'], app.config['ARCANE_OUTPUT_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
PROCESSED_FOLDER = os.path.join(os.getcwd(), 'anime_output')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

if not os.path.exists(app.config['ANIME_OUTPUT_FOLDER']):
    os.makedirs(app.config['ANIME_OUTPUT_FOLDER'])

if not os.path.exists(app.config['CYBERPUNK_OUTPUT_FOLDER']):
    os.makedirs(app.config['CYBERPUNK_OUTPUT_FOLDER'])

if not os.path.exists(app.config['ARCANE_OUTPUT_FOLDER']):
    os.makedirs(app.config['ARCANE_OUTPUT_FOLDER'])

for folder in [app.config['UPLOAD_FOLDER'], app.config['SAVED_IMAGES_FOLDER'], app.config['CYBERPUNK_OUTPUT_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
PROCESSED_FOLDER = os.path.join(os.getcwd(), 'cyberpunk_output')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

for folder in [app.config['UPLOAD_FOLDER'], app.config['SAVED_IMAGES_FOLDER'], app.config['ARCANE_OUTPUT_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
PROCESSED_FOLDER = os.path.join(os.getcwd(), 'arcane_output')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# Initialize the ONNX model session
model_path = "model/AnimeGANv2/AnimeGANv2-master/pb_and_onnx_model/Shinkai_53.onnx"
anime_model = load_model(model_path)

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("Error loading face cascade classifier")

@app.route('/')
def index():
    saved_images = os.listdir(SAVED_IMAGES_FOLDER)  # List saved images for selection
    return render_template('frontend.html', saved_images=saved_images)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Generate a unique filename based on the current timestamp
    filename = f"{int(time.time())}_{file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(file_path)
    except Exception as e:
        print(f"Error saving image: {e}")
        return jsonify({"error": "Failed to save image"}), 500

    return jsonify({'filename': filename}), 200

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/capture', methods=['POST'])
def capture_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file part in the request"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the captured image with a unique filename
    filename = f"captured_image_{int(time.time())}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        image.save(filepath)
        print(f"Image saved at {filepath}")  # Debugging log
    except Exception as e:
        print(f"Error saving image: {e}")
        return jsonify({"error": "Failed to save image"}), 500

    return jsonify({"filename": filename}), 200

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    b = int(hex_color[4:6], 16)
    g = int(hex_color[2:4], 16)
    r = int(hex_color[0:2], 16)
    return (b, g, r)

@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    filename = data.get('filename')
    effect = data.get('effect')
    color = data.get('color')
    background_image = data.get('background_image')

    img_path = os.path.join(UPLOAD_FOLDER, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Could not read image at {img_path}")
        return jsonify({"error": "Image not found"}), 404

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    print(f"Detected faces: {faces}")
    face_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # Initialize processed_filename
    processed_filename = None
    # Apply selected effects
    if effect == 'blur':
        for (x, y, w, h) in faces:
            if w > 0 and h > 0:  # Check if width and height are valid
                face_region = img[y:y + h, x:x + w]
                blurred_face = cv2.GaussianBlur(face_region, (15, 15), 0)
                img[y:y + h, x:x + w] = blurred_face
            else:
                print("Invalid face region dimensions")

    elif effect == 'frame':
        frame_thickness = 10
        img = cv2.copyMakeBorder(img, frame_thickness, frame_thickness, frame_thickness, frame_thickness,
                                  cv2.BORDER_CONSTANT, value=[0, 255, 0])  # Green frame
    if effect == 'crop':
        # Check if at least one face is detected
        if len(faces) > 0:
            # Get the largest face detected (in case of multiple faces)
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            # Crop the image to the face region
            img = img[y:y + h, x:x + w]
        else:
            print("No faces found for cropping.")

    if effect == 'color' and color:
        bgr_color = hex_to_bgr(color)
        overlay = img.copy()
        overlay[:] = bgr_color
        img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)  # Adjust blending as needed

    elif effect == 'background' and background_image:
        bg_path = os.path.join(SAVED_IMAGES_FOLDER, background_image)
        background = cv2.imread(bg_path)

        if background is None:
            return jsonify({"error": "Background image not found"}), 404

        # Resize background to match input image dimensions
        background = cv2.resize(background, (img.shape[1], img.shape[0]))
        # Convert image to grayscale and detect face
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Create a mask for background replacement
        mask = np.zeros_like(gray)
        for (x, y, w, h) in faces:
            # Define an extended region around the face for the mask
            extended_x = max(0, x - int(w * 0.3))
            extended_y = max(0, y - int(h * 0.3))
            extended_w = min(img.shape[1], x + w + int(w * 0.3)) - extended_x
            extended_h = min(img.shape[0], y + h + int(h * 0.3)) - extended_y
            # Draw an elliptical mask for smoother transitions
            center = (x + w // 2, y + h // 2)
            axes = (extended_w // 2, extended_h // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        # Feather the mask edges to blend the face area smoothly with the new background
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        # Separate the face and background areas
        img_face = cv2.bitwise_and(img, img, mask=mask)  # Face area only
        img_bg_replaced = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))  # Background area only
        # Combine face and new background with adjusted weights to ensure balance
        img = cv2.addWeighted(img_bg_replaced, 0.7, img_face, 1.0, 0)

    # Save processed image
    processed_filename = f"processed_{int(time.time())}.jpg"
    processed_path = os.path.join(UPLOAD_FOLDER, processed_filename)
    cv2.imwrite(processed_path, img)
    if not os.path.exists(processed_path):
        print(f"Error: Processed image not found at {processed_path}")

    return jsonify({'processed_filename': processed_filename}), 200

@app.route('/save', methods=['POST'])
def save_image():
    data = request.json
    filename = data.get('filename')

    if not filename:
        return jsonify({'error': 'Filename not provided'}), 400

    # Define paths
    file_path_upload = os.path.join(UPLOAD_FOLDER, filename)
    file_path_anime = os.path.join(ANIME_OUTPUT_FOLDER, filename)
    file_path_cyberpunk = os.path.join(CYBERPUNK_OUTPUT_FOLDER, filename)
    file_path_arcane = os.path.join(ARCANE_OUTPUT_FOLDER, filename)
    file_path_saved = os.path.join(SAVED_IMAGES_FOLDER, filename)

    response = {}

    # Try to move the file from UPLOAD_FOLDER to SAVED_IMAGES_FOLDER
    if os.path.exists(file_path_upload):
        try:
            os.rename(file_path_upload, file_path_saved)
            response['moved_from'] = 'UPLOAD_FOLDER'
        except Exception as e:
            print(f"Error moving image from upload to saved folder: {e}")
            return jsonify({'error': 'Failed to move image from upload folder to saved folder'}), 500

    # If not in UPLOAD_FOLDER, try moving it from ANIME_OUTPUT_FOLDER to SAVED_IMAGES_FOLDER
    elif os.path.exists(file_path_anime):
        try:
            os.rename(file_path_anime, file_path_saved)
            response['moved_from'] = 'ANIME_OUTPUT_FOLDER'
        except Exception as e:
            print(f"Error moving image from anime output to saved folder: {e}")
            return jsonify({'error': 'Failed to move image from anime output folder to saved folder'}), 500
    elif os.path.exists(file_path_cyberpunk):
        try:
            os.rename(file_path_cyberpunk, file_path_saved)
            response['moved_from'] = 'CYBERPUNK_OUTPUT_FOLDER'
        except Exception as e:
            print(f"Error moving image from cyberpunk output to saved folder: {e}")
            return jsonify({'error': 'Failed to move image from cyberpunk output folder to saved folder'}), 500
    elif os.path.exists(file_path_arcane):
        try:
            os.rename(file_path_arcane, file_path_saved)
            response['moved_from'] = 'ARCANE_OUTPUT_FOLDER'
        except Exception as e:
            print(f"Error moving image from arcane output to saved folder: {e}")
            return jsonify({'error': 'Failed to move image from arcane output folder to saved folder'}), 500

    else:
        # If file is in neither folder
        return jsonify({'error': 'File not found in either folder'}), 404

    return jsonify(response), 200


@app.route('/delete', methods=['POST'])
def delete_image():
    data = request.json
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'Filename not provided'}), 400

    file_path_upload = os.path.join(UPLOAD_FOLDER, filename)
    file_path_anime = os.path.join(ANIME_OUTPUT_FOLDER, filename)
    file_path_cyberpunk = os.path.join(CYBERPUNK_OUTPUT_FOLDER, filename)
    file_path_arcane = os.path.join(ARCANE_OUTPUT_FOLDER, filename)
    response = {}

    # Attempt to delete file from UPLOAD_FOLDER
    if os.path.exists(file_path_upload):
        try:
            os.remove(file_path_upload)
            response['deleted_from_upload'] = filename
        except Exception as e:
            print(f"Error deleting image from upload folder: {e}")
            response['error_upload'] = 'Failed to delete from upload folder'

    # Attempt to delete file from ANIME_OUTPUT_FOLDER
    if os.path.exists(file_path_anime):
        try:
            os.remove(file_path_anime)
            response['deleted_from_anime'] = filename
        except Exception as e:
            print(f"Error deleting image from anime output folder: {e}")
            response['error_anime'] = 'Failed to delete from anime output folder'

    # Attempt to delete file from CYBERPUNK_OUTPUT_FOLDER
    if os.path.exists(file_path_cyberpunk):
        try:
            os.remove(file_path_cyberpunk)
            response['deleted_from_cyberpunk'] = filename
        except Exception as e:
            print(f"Error deleting image from cyberpunk output folder: {e}")
            response['error_cyberpunk'] = 'Failed to delete from cyberpunk output folder'
    # Attempt to delete file from ARCANE_OUTPUT_FOLDER
    if os.path.exists(file_path_arcane):
        try:
            os.remove(file_path_arcane)
            response['deleted_from_arcane'] = filename
        except Exception as e:
            print(f"Error deleting image from arcane output folder: {e}")
            response['error_cyberpunk'] = 'Failed to delete from arcane output folder'
    # If file wasn't found in any folder
    if 'deleted_from_upload' not in response and 'deleted_from_anime' not in response and 'deleted_from_cyberpunk' not in response and 'deleted_from_arcane' not in response:
        response['error'] = 'File not found in any folder'
        return jsonify(response), 404

    return jsonify(response), 200

@app.route('/saved_images', methods=['GET'])
def get_saved_images():
    try:
        saved_images = os.listdir(SAVED_IMAGES_FOLDER)  # Fixed path
        return jsonify({'saved_images': saved_images})
    except Exception as e:
        print(f"Error retrieving saved images: {e}")
        return jsonify({'error': 'Could not retrieve images'}), 500

@app.route('/saved_images/<filename>')
def saved_image(filename):
    return send_from_directory(SAVED_IMAGES_FOLDER, filename)

@app.route('/process_anime', methods=['POST'])
def process_anime_image():
    data = request.json
    filename = data.get('filename')
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(img_path):
        return jsonify({"error": "Image not found"}), 404

    try:
        transform_images(model_path=model_path, image_folder=app.config['UPLOAD_FOLDER'], output_folder=app.config['ANIME_OUTPUT_FOLDER'])
        output_filename = f"anime_{filename}"  # Ensure this matches the filename structure you expect
        return jsonify({'processed_filename': output_filename}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

@app.route('/anime_output/<filename>')
def anime_output_file(filename):
    return send_from_directory(app.config['ANIME_OUTPUT_FOLDER'], filename)

def load_cyberpunk_model(model_id="DGSpitzer/Cyberpunk-Anime-Diffusion", token="HUGGING_FACE_TOKEN"):
    """Load the Cyberpunk diffusion model."""
    login(token)  # Ensure login happens here
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    return pipe

@app.route('/process_cyberpunk', methods=['POST'])
def process_cyberpunk_image():
    data = request.json
    filename = data.get('filename')
    prompt = data.get('prompt')
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not filename or not os.path.exists(img_path):
        return jsonify({"error": "Image not found"}), 404

    try:
        # Ensure the transform function parameters match the intended use for cyberpunk transformation
        transform_images_cyberpunk(
            model_dir=app.config.get('CYBERPUNK_MODEL_PATH', 'DGSpitzer/Cyberpunk-Anime-Diffusion'),
            image_folder=app.config['UPLOAD_FOLDER'],
            output_folder=app.config['CYBERPUNK_OUTPUT_FOLDER'],
            prompt = prompt
        )

        # Generate a filename for the processed image to ensure it matches the format you expect
        output_filename = f"cyberpunk_{filename}"
        return jsonify({'processed_filename': output_filename}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

@app.route('/cyberpunk_output/<filename>')
def serve_cyberpunk_image(filename):
    # Ensure the correct folder is being served
    return send_from_directory(app.config['CYBERPUNK_OUTPUT_FOLDER'], filename)

def load_arcane_model(model_id="nitrosocke/Arcane-Diffusion", token="HUGGING_FACE_TOKEN"):
    """Load the Arcane diffusion model."""
    login(token)  # Ensure login happens here
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    return pipe

@app.route('/process_arcane', methods=['POST'])
def process_arcane_image():
    data = request.json
    filename = data.get('filename')
    prompt = data.get('prompt')
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not filename or not os.path.exists(img_path):
        return jsonify({"error": "Image not found"}), 404

    try:
        # Ensure the transform function parameters match the intended use for arcane transformation
        transform_images_arcane(
            model_dir=app.config.get('ARCANE_MODEL_PATH', 'nitrosocke/Arcane-Diffusion'),
            image_folder=app.config['UPLOAD_FOLDER'],
            output_folder=app.config['ARCANE_OUTPUT_FOLDER'],
            prompt = prompt
        )

        # Generate a filename for the processed image to ensure it matches the format you expect
        output_filename = f"arcane_{filename}"
        return jsonify({'processed_filename': output_filename}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

@app.route('/arcane_output/<filename>')
def serve_arcane_image(filename):
    # Ensure the correct folder is being served
    return send_from_directory(app.config['ARCANE_OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
