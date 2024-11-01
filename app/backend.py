from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import time
import cv2
import numpy as np

app = Flask(__name__)

# Configuration for upload folders
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Directory for saved images
SAVED_IMAGES_FOLDER = os.path.join(os.getcwd(), 'saved_images')
if not os.path.exists(SAVED_IMAGES_FOLDER):
    os.makedirs(SAVED_IMAGES_FOLDER)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
        return jsonify({"error": "Image not found"}), 404

    if isinstance(background_image, list) and len(background_image) > 0:
        background_image = background_image[0]

    # Apply selected effects
    if effect == 'blur':
        # Detect face and apply blur only to the face
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face_region = img[y:y + h, x:x + w]
            blurred_face = cv2.GaussianBlur(face_region, (15, 15), 0)
            img[y:y + h, x:x + w] = blurred_face  # Replace face region with blurred version

    elif effect == 'frame':
        frame_thickness = 10
        img = cv2.copyMakeBorder(img, frame_thickness, frame_thickness, frame_thickness, frame_thickness,
                                  cv2.BORDER_CONSTANT, value=[0, 255, 0])  # Green frame
    elif effect == 'crop':
        h, w, _ = img.shape
        start_row, start_col = int(h * 0.1), int(w * 0.1)
        end_row, end_col = int(h * 0.9), int(w * 0.9)
        img = img[start_row:end_row, start_col:end_col]
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

    return jsonify({'processed_filename': processed_filename}), 200

@app.route('/save', methods=['POST'])
def save_image():
    data = request.json
    filename = data.get('filename')

    if not filename:
        return jsonify({'error': 'Filename not provided'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    saved_path = os.path.join(SAVED_IMAGES_FOLDER, filename)

    try:
        os.rename(file_path, saved_path)
    except Exception as e:
        print(f"Error saving image: {e}")
        return jsonify({'error': 'Failed to save image'}), 500

    return jsonify({'saved_filename': filename}), 200

@app.route('/delete', methods=['POST'])
def delete_image():
    data = request.json
    filename = data.get('filename')

    if not filename:
        return jsonify({'error': 'Filename not provided'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, filename)

    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return jsonify({'deleted_filename': filename}), 200
        except Exception as e:
            print(f"Error deleting image: {e}")
            return jsonify({'error': 'Failed to delete image'}), 500

    return jsonify({'error': 'File not found'}), 404

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

if __name__ == '__main__':
    app.run(debug=True)
