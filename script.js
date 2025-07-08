// Global variables
let video, canvas, ctx;
let isRunning = false;
let faceDetection;
let emotionModel;
let gifElements = {};
let animationId;

// Emotion labels (same order as your model)
const emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"];

// Emotion colors for bounding boxes
const emotionColors = {
    "Neutral": [255, 255, 255],
    "Happy": [0, 255, 255],
    "Surprise": [0, 165, 255],
    "Sad": [255, 0, 0],
    "Angry": [0, 0, 255],
    "Disgust": [128, 0, 128],
    "Fear": [255, 255, 0],
    "Contempt": [0, 255, 0]
};

// Emotion text colors (cycling colors)
const emotionTextColors = {
    "Neutral": [[255,255,255], [224,212,196], [228,203,179]],
    "Happy": [[182,110,68], [76,235,253], [83,169,242]],
    "Surprise": [[247,255,0], [42,42,165], [232,206,0]],
    "Sad": [[194,105,3], [228,172,32], [237,202,162]],
    "Angry": [[61, 57, 242], [49,121,249], [232,220,214]],
    "Disgust": [[70,190,77], [120,159,6], [100,55,124]],
    "Fear": [[198, 128, 134], [133,71,68], [80,45,98]],
    "Contempt": [[160, 134, 72], [145, 180, 250], [173, 217, 251]]
};

let colorIndex = 0;
let frameCount = 0;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Page loaded, initializing...');
    
    // Get DOM elements
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const loading = document.getElementById('loading');
    
    // Add button event listeners
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    
    try {
        loading.textContent = 'Loading face detection...';
        await initializeFaceDetection();
        
        loading.textContent = 'Loading emotion model...';
        await loadEmotionModel();
        
        loading.textContent = 'Loading GIFs...';
        await loadGIFs();
        
        loading.textContent = 'Ready! Click Start Camera';
        console.log('All models loaded successfully!');
        
    } catch (error) {
        console.error('Initialization error:', error);
        loading.textContent = 'Error loading models. Check console.';
    }
});

// Initialize MediaPipe Face Detection
async function initializeFaceDetection() {
    faceDetection = new FaceDetection({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection@0.4.1646425229/${file}`;
        }
    });
    
    faceDetection.setOptions({
        model: 'short',
        minDetectionConfidence: 0.5
    });
    
    await faceDetection.initialize();
    console.log('Face detection initialized');
}

// Load ONNX emotion model
async function loadEmotionModel() {
    try {
        // Create ONNX inference session
        emotionModel = await ort.InferenceSession.create('./models/emotion_model.onnx');
        console.log('Emotion model loaded successfully');
        console.log('Model inputs:', emotionModel.inputNames);
        console.log('Model outputs:', emotionModel.outputNames);
    } catch (error) {
        console.error('Error loading emotion model:', error);
        throw error;
    }
}

// Load GIF animations
async function loadGIFs() {
    const gifPromises = emotions.map(emotion => {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                // Create a canvas to hold the animated GIF
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                
                gifElements[emotion] = img; // Keep the original image for animation
                resolve();
            };
            img.onerror = () => {
                console.warn(`Could not load GIF for ${emotion}`);
                resolve(); // Don't fail, just continue without this GIF
            };
            img.src = `./static/${emotion}.gif`;
        });
    });
    
    await Promise.all(gifPromises);
    console.log('GIFs loaded:', Object.keys(gifElements));
}

// Start camera
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { 
                width: 640, 
                height: 480,
                facingMode: 'user'
            }
        });
        
        video.srcObject = stream;
        await video.play();
        
        // Set canvas size to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        isRunning = true;
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        document.getElementById('loading').classList.add('hidden');
        
        // Start processing
        processVideo();
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Could not access camera. Please check permissions.');
    }
}

// Stop camera
function stopCamera() {
    isRunning = false;
    
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }
    
    if (animationId) {
        cancelAnimationFrame(animationId);
    }
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('loading').textContent = 'Ready! Click Start Camera';
}

// Main video processing loop
async function processVideo() {
    if (!isRunning) return;
    
    try {
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Update frame counter and color cycling
        frameCount++;
        if (frameCount % 10 === 0) {
            colorIndex = (colorIndex + 1) % 3;
        }
        
        // Detect faces
        const faces = await detectFaces();
        
        // Process each face
        for (const face of faces) {
            await processFace(face);
        }
        
    } catch (error) {
        console.error('Error in video processing:', error);
    }
    
    // Continue processing
    animationId = requestAnimationFrame(processVideo);
}

// Detect faces using MediaPipe
async function detectFaces() {
    return new Promise((resolve) => {
        // Create a canvas to get image data
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(video, 0, 0);
        
        // Convert to MediaPipe format
        faceDetection.onResults((results) => {
            const faces = [];
            if (results.detections) {
                for (const detection of results.detections) {
                    const bbox = detection.boundingBox;
                    faces.push({
                        x: bbox.xCenter * canvas.width - (bbox.width * canvas.width) / 2,
                        y: bbox.yCenter * canvas.height - (bbox.height * canvas.height) / 2,
                        width: bbox.width * canvas.width,
                        height: bbox.height * canvas.height
                    });
                }
            }
            resolve(faces);
        });
        
        faceDetection.send({image: tempCanvas});
    });
}

// Process individual face
async function processFace(face) {
    // Extract face region from video
    const faceCanvas = document.createElement('canvas');
    faceCanvas.width = 48;
    faceCanvas.height = 48;
    const faceCtx = faceCanvas.getContext('2d');
    
    // Draw face region and convert to grayscale
    faceCtx.drawImage(
        video,
        face.x, face.y, face.width, face.height,
        0, 0, 48, 48
    );
    
    // Get image data and convert to grayscale
    const imageData = faceCtx.getImageData(0, 0, 48, 48);
    const grayData = new Float32Array(48 * 48);
    
    for (let i = 0; i < imageData.data.length; i += 4) {
        const gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
        grayData[i / 4] = (gray / 255.0 - 0.5) / 0.5; // Normalize to [-1, 1]
    }
    
    // Reshape for model input [1, 1, 48, 48]
    const inputTensor = new ort.Tensor('float32', grayData, [1, 1, 48, 48]);
    
    try {
        // Run emotion model
        const results = await emotionModel.run({ [emotionModel.inputNames[0]]: inputTensor });
        const predictions = results[emotionModel.outputNames[0]].data;
        
        // Find top emotion
        let maxIdx = 0;
        let maxProb = predictions[0];
        for (let i = 1; i < predictions.length; i++) {
            if (predictions[i] > maxProb) {
                maxProb = predictions[i];
                maxIdx = i;
            }
        }
        
        const topEmotion = emotions[maxIdx];
        
        // Draw results
        drawFaceResults(face, predictions, topEmotion);
        
    } catch (error) {
        console.error('Error running emotion model:', error);
    }
}

// Draw face detection and emotion results
function drawFaceResults(face, predictions, topEmotion) {
    // Draw bounding box
    const color = emotionColors[topEmotion];
    ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    ctx.lineWidth = 2;
    ctx.strokeRect(face.x, face.y, face.width, face.height);
    
    // Draw animated GIF if available
    if (gifElements[topEmotion]) {
        const gifSize = Math.min(face.width, face.height) / 2;
        const gifX = face.x + face.width / 2 - gifSize / 2;
        const gifY = face.y - gifSize - 10;
        
        // For animated GIFs, we need to create a img element and append it temporarily
        // But since canvas doesn't support GIF animation, we'll use a different approach
        // We'll create an overlay div with the GIF
        drawAnimatedGIF(topEmotion, gifX, gifY, gifSize);
    }
    
    // Draw emotion probabilities
    const textStartY = face.y - (gifElements[topEmotion] ? Math.min(face.width, face.height) / 2 + 30 : 20);
    
    emotions.forEach((emotion, i) => {
        const prob = Math.round(predictions[i] * 100);
        const text = `${emotion}: ${prob}%`;
        
        let textColor;
        if (emotion === topEmotion) {
            const colors = emotionTextColors[topEmotion][colorIndex];
            textColor = `rgb(${colors[0]}, ${colors[1]}, ${colors[2]})`;
        } else {
            textColor = 'rgb(255, 255, 255)';
        }
        
        ctx.fillStyle = textColor;
        ctx.font = '14px Arial';
        ctx.fillText(text, face.x, textStartY - (i * 18));
    });
}

// Draw animated GIF overlay
function drawAnimatedGIF(emotion, x, y, size) {
    // Remove any existing GIF overlays
    const existingGifs = document.querySelectorAll('.gif-overlay');
    existingGifs.forEach(gif => gif.remove());
    
    // Create new GIF overlay
    const gifOverlay = document.createElement('img');
    gifOverlay.src = `./static/${emotion}.gif`;
    gifOverlay.className = 'gif-overlay';
    gifOverlay.style.position = 'absolute';
    gifOverlay.style.left = (canvas.offsetLeft + x) + 'px';
    gifOverlay.style.top = (canvas.offsetTop + y) + 'px';
    gifOverlay.style.width = size + 'px';
    gifOverlay.style.height = size + 'px';
    gifOverlay.style.pointerEvents = 'none';
    gifOverlay.style.zIndex = '1000';
    
    // Add to video container
    document.querySelector('.video-container').appendChild(gifOverlay);
    
    // Remove after a short time to prevent accumulation
    setTimeout(() => {
        if (gifOverlay.parentNode) {
            gifOverlay.remove();
        }
    }, 100);
}