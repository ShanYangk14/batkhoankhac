 document.addEventListener('DOMContentLoaded', function() {
        const uploadForm = document.getElementById('uploadForm');
        const uploadedImageDiv = document.getElementById('uploadedImage');
        const captureBtn = document.getElementById('captureBtn');
        const video = document.getElementById('video');
        const brightnessSlider = document.getElementById('brightness');
        const contrastSlider = document.getElementById('contrast');
        const brightnessValueDisplay = document.getElementById('brightnessValue');
        const contrastValueDisplay = document.getElementById('contrastValue');
        const saveBtn = document.getElementById('saveBtn');
        const deleteBtn = document.getElementById('deleteBtn');
        const effectSelect = document.getElementById('effect');
        const applyEffectBtn = document.getElementById('applyEffect');
        const colorPicker = document.getElementById('colorPicker');
        const backgroundContainer = document.getElementById('backgroundContainer');
        const selectBackgroundBtn = document.getElementById('selectBackgroundBtn');
        const colorOverlay = document.getElementById('colorOverlay');
        const resetBtn = document.getElementById('resetBtn');
        const backgroundSelector = document.getElementById('backgroundSelector');
        const promptInput = document.getElementById('promptInput');
        let selectedImages = [];
        let previousImages = [];
        let originalImageSrc = null;

        promptInput.style.display = 'none';
         // Add "Anime Style Transfer" to the effect dropdown
        const animeOption = document.createElement('option');
        animeOption.value = 'anime';
        effectSelect.appendChild(animeOption);
        // Access the camera for live preview
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
            })
            .catch(error => {
                console.error('Error accessing camera:', error);
            });
            function updateVideoFilters() {
            const brightness = brightnessSlider.value;
            const contrast = contrastSlider.value;
            video.style.filter = `brightness(${parseFloat(brightness) / 100 + 1}) contrast(${contrast})`;
            brightnessValueDisplay.textContent = brightness;
            contrastValueDisplay.textContent = contrast;
        }
        function updateFiltersForImage(imageElement) {
            const brightness = brightnessSlider.value;
            const contrast = contrastSlider.value;
            imageElement.style.filter = `brightness(${parseFloat(brightness) / 100 + 1}) contrast(${contrast})`;
        }

        // Event listeners for brightness and contrast sliders
        brightnessSlider.addEventListener('input', updateVideoFilters);
        contrastSlider.addEventListener('input', updateVideoFilters);

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

       captureBtn.addEventListener('click', async () => {
            // Create a canvas and set its dimensions to match the video
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');

            // Get the current brightness and contrast values
            const brightness = brightnessSlider.value;
            const contrast = contrastSlider.value;

            // Apply the filters to the canvas context
            ctx.filter = `brightness(${parseFloat(brightness) / 100 + 1}) contrast(${contrast})`;

            // Draw the video frame to the canvas with the applied filters
            ctx.drawImage(video, 0, 0);

            // Convert the canvas to a data URL and send it to the server
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

     // Apply brightness and contrast to the displayed image
    updateFiltersForImage(imgElement);

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
    const prompt = document.getElementById('promptInput').value;
    const color = (effect === 'color') ? colorPicker.value : null;
    const backgroundImage = selectedImages.length > 0 ? selectedImages[0] : null;

   try {
        if (effect === 'cyberpunk') {
            response = await fetch('/process_cyberpunk', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ filename, prompt })
            });
        } else if (effect == 'arcane'){
                  response = await fetch('/process_arcane', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ filename, prompt })
            });
        } else if (effect === 'anime') {
            response = await fetch('/process_anime', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ filename })
            });
        } else {
            // Use existing process endpoint for other effects
            response = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ filename, effect, color, background_image: backgroundImage })
            });
        }

        const data = await response.json();
        console.log(data);
      if (response.ok) {
         uploadedImageDiv.innerHTML = '';  // Clear previous images

            // Define paths based on effect
            let processedImgSrc;
            if (effect === 'cyberpunk') {
                processedImgSrc = `/cyberpunk_output/${data.processed_filename}`;
            } else if(effect === 'arcane'){
                processedImgSrc = `/arcane_output/${data.processed_filename}`;
            } else if (effect === 'anime') {
                processedImgSrc = `/anime_output/${data.processed_filename}`;
            } else {
                processedImgSrc = `/uploads/${data.processed_filename}`;
            }

            // Append the processed image
            const processedImg = document.createElement('img');
            processedImg.src = processedImgSrc;
            uploadedImageDiv.appendChild(processedImg);

             // Apply brightness and contrast to processed image
            updateFiltersForImage(processedImg);

             // Save original image if it's the first effect
            if (previousImages.length === 0 && !originalImageSrc) {
                originalImageSrc = imgElement.src; // Set original image
            }

            // Push the processed image to previousImages
            previousImages.push(processedImgSrc);
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        console.error('Error applying effect:', error);
    }
});

resetBtn.addEventListener('click', () => {
    if (previousImages.length > 0) {
        previousImages.pop(); // Remove the current image from history
        const lastImageSrc = previousImages.length > 0 ? previousImages[previousImages.length - 1] : originalImageSrc; // Get the previous image or original image
        displayUploadedImage(lastImageSrc); // Restore the previous image or original

        // Clear selections
        selectedImages = [];
        Array.from(backgroundContainer.children).forEach(img => {
            img.style.border = 'none'; // Remove selection border
        });
    } else if (originalImageSrc) {
        // If no history, reset to the original image if it exists
        displayUploadedImage(originalImageSrc);

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
                backgroundContainer.style.display = 'none';
            } else if (effectSelect.value === 'background') {
                colorPicker.style.display = 'none';
                backgroundContainer.style.display = 'block';
            } else if (effectSelect.value === 'anime') {
                colorPicker.style.display = 'none';
                backgroundContainer.style.display = 'none';
            } else {
                colorPicker.style.display = 'none';
                backgroundContainer.style.display = 'none';
            }
        });

        effectSelect.addEventListener('change', function() {
        const selectedEffect = effectSelect.value;

        // Show the prompt input if "cyberpunk" or "arcane" is selected
        if (selectedEffect === 'cyberpunk' || selectedEffect === 'arcane') {
            promptInput.style.display = 'block';  // Show the prompt input
        } else {
            promptInput.style.display = 'none';  // Hide the prompt input
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
});