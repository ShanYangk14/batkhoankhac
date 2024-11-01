  const uploadForm = document.getElementById('uploadForm');
        const uploadedImageDiv = document.getElementById('uploadedImage');
        const captureBtn = document.getElementById('captureBtn');
        const video = document.getElementById('video');
        const saveBtn = document.getElementById('saveBtn');
        const deleteBtn = document.getElementById('deleteBtn');
        const effectSelect = document.getElementById('effect');
        const applyEffectBtn = document.getElementById('applyEffect');
        const colorPicker = document.getElementById('colorPicker');
        const backgroundContainer = document.getElementById('backgroundContainer');
        const selectBackgroundBtn = document.getElementById('selectBackgroundBtn');
        const colorOverlay = document.getElementById('colorOverlay');
        const resetBtn = document.getElementById('resetBtn');
        let selectedImages = [];
        let previousImages = [];
        let originalImageSrc = null;

        // Access the camera for live preview
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
            })
            .catch(error => {
                console.error('Error accessing camera:', error);
            });

        // Handle file upload
        uploadForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const formData = new FormData(uploadForm);
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (response.ok) {
                    const img = document.createElement('img');
                    img.src = `/uploads/${data.filename}`;
                    uploadedImageDiv.innerHTML = ''; // Clear previous image
                    uploadedImageDiv.appendChild(img);
                      uploadedImg.style.display = 'block';
                    colorOverlay.style.backgroundColor = 'transparent';
                } else {
                    alert(`Error: ${data.error}`);
                }
            } catch (error) {
                console.error('Error uploading image:', error);
            }
        });

        // Capture image from video stream
        captureBtn.addEventListener('click', async () => {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);

            const dataUrl = canvas.toDataURL('image/png');
            const blob = await fetch(dataUrl).then(res => res.blob());

            const formData = new FormData();
            formData.append('image', blob, 'captured_image.png');

            try {
                const response = await fetch('/capture', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (response.ok) {
                    const img = document.createElement('img');
                    img.src = `/uploads/${data.filename}`;
                    uploadedImageDiv.innerHTML = ''; // Clear previous image
                    uploadedImageDiv.appendChild(img);
                } else {
                    alert(`Error: ${data.error}`);
                }
            } catch (error) {
                console.error('Error capturing image:', error);
            }
        });

        // Save image
        saveBtn.addEventListener('click', async () => {
            const imgElement = uploadedImageDiv.querySelector('img');
            if (!imgElement) {
                alert('No image to save');
                return;
            }
            const filename = imgElement.src.split('/').pop();
            try {
                const response = await fetch('/save', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ filename })
                });
                const data = await response.json();
                if (response.ok) {
                    alert(`Image saved as: ${data.saved_filename}`);
                } else {
                    alert(`Error: ${data.error}`);
                }
            } catch (error) {
                console.error('Error saving image:', error);
            }
        });

        // Delete image
        deleteBtn.addEventListener('click', async () => {
            const imgElement = uploadedImageDiv.querySelector('img');
            if (!imgElement) {
                alert('No image to delete');
                return;
            }
            const filename = imgElement.src.split('/').pop();
            try {
                const response = await fetch('/delete', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ filename })
                });
                const data = await response.json();
                if (response.ok) {
                    alert(`Image deleted: ${data.deleted_filename}`);
                    uploadedImageDiv.innerHTML = ''; // Clear displayed image
                } else {
                    alert(`Error: ${data.error}`);
                }
            } catch (error) {
                console.error('Error deleting image:', error);
            }
        });
const displayUploadedImage = (imageSrc) => {
    uploadedImageDiv.innerHTML = ''; // Clear previous image
    const imgElement = document.createElement('img');
    imgElement.src = imageSrc;
    uploadedImageDiv.appendChild(imgElement);

    // Store previous images
    previousImages.push(imgElement.src); // Track displayed images
};
        // Apply selected effect
  applyEffectBtn.addEventListener('click', async () => {
    const imgElement = uploadedImageDiv.querySelector('img');
    if (!imgElement) {
        alert('No image to apply effect to');
        return;
    }
    const filename = imgElement.src.split('/').pop();
    const effect = effectSelect.value;
    const color = (effect === 'color') ? colorPicker.value : null;
    const backgroundImage = selectedImages.length > 0 ? selectedImages[0] : null;

    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ filename, effect, color, background_image: backgroundImage })
        });
        const data = await response.json();
        if (response.ok) {
            const processedImg = document.createElement('img');
            processedImg.src = `/uploads/${data.processed_filename}`;
            uploadedImageDiv.innerHTML = '';
            uploadedImageDiv.appendChild(processedImg);
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        console.error('Error applying effect:', error);
    }
});

resetBtn.addEventListener('click', () => {
            if (previousImages.length > 1) {
                previousImages.pop(); // Remove the current image
                const lastImageSrc = previousImages[previousImages.length - 1]; // Get the last image
                displayUploadedImage(lastImageSrc); // Restore the last image

                // Clear selections
                selectedImages = [];
                Array.from(backgroundContainer.children).forEach(img => {
                    img.style.border = 'none'; // Remove selection border
                });
            } else {
                alert('No previous image to reset to!');
            }
        });

        // Load saved images into selector
        async function loadSavedImages() {
            try {
                const response = await fetch('/saved_images');
                const data = await response.json();
                if (response.ok) {
                    backgroundSelector.innerHTML = ''; // Clear previous options
                    data.saved_images.forEach(image => {
                        const option = document.createElement('option');
                        option.value = image;
                        option.textContent = image;
                        backgroundSelector.appendChild(option);
                    });
                } else {
                    console.error('Error loading saved images:', data.error);
                }
            } catch (error) {
                console.error('Error fetching saved images:', error);
            }
        }

        // Show/Hide Select Background Button based on effect selection
        effectSelect.addEventListener('change', () => {
            if (effectSelect.value === 'background') {
                selectBackgroundBtn.style.display = 'block'; // Show button
            } else {
                selectBackgroundBtn.style.display = 'none'; // Hide button
            }
        });
        // Show/hide color picker based on selected effect
        effectSelect.addEventListener('change', () => {
            if (effectSelect.value === 'color') {
                colorPicker.style.display = 'block';
                backgroundContainer.style.display = 'none'; // Hide background selector
            } else if (effectSelect.value === 'background') {
                colorPicker.style.display = 'none'; // Hide color picker
                backgroundContainer.style.display = 'block';
            } else {
                colorPicker.style.display = 'none'; // Hide color picker
                backgroundContainer.style.display = 'none'; // Hide background selector
            }
        });

        // Randomly select a background image from saved images
selectBackgroundBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/saved_images');
        const data = await response.json();

        if (response.ok) {
            const savedImages = data.saved_images;
            backgroundContainer.innerHTML = ''; // Clear existing content

            // Display images in backgroundContainer
            savedImages.forEach(image => {
                const imgElement = document.createElement('img');
                imgElement.src = `/saved_images/${image}`;
                imgElement.alt = image;
                imgElement.style.width = '100px';
                imgElement.style.margin = '5px';
                imgElement.style.cursor = 'pointer';

                // Toggle selection of each background image
                imgElement.addEventListener('click', () => {
                    if (selectedImages.includes(image)) {
                        // Deselect image if already selected
                        imgElement.style.border = 'none';
                        selectedImages = selectedImages.filter(img => img !== image);
                    } else {
                        // Select the clicked image
                        imgElement.style.border = '2px solid blue';
                        selectedImages.push(image);
                    }
                });

                backgroundContainer.appendChild(imgElement);
            });
        } else {
            console.error('Error loading saved images:', data.error);
        }
    } catch (error) {
        console.error('Error fetching saved images:', error);
    }
});

    const processImage = async () => {
    const effect = 'background';  // Set the desired effect
    const backgroundImage = backgroundSelector.value;  // Get selected background image

    const response = await fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            filename: yourImageFilename,  // Pass your uploaded image's filename
            effect: effect,
            background_image: backgroundImage  // Pass the selected background image
        })
    });

    const result = await response.json();
    if (response.ok) {
        console.log('Processed image:', result.processed_filename);
    } else {
        console.error('Error processing image:', result.error);
    }
};
        loadSavedImages(); // Call to load saved images on page load
