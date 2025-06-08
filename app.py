from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from models import PerformanceModel
import base64
import io
from PIL import Image, ImageSequence
import json
import os


app = Flask(__name__)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PerformanceModel(n_classes=8, logits=False)
model.load_state_dict(torch.load('ferplus_model_pd_acc.pth', map_location=device))
model.to(device)
model.eval()

# Ensure model is in eval mode for consistent dropout/batchnorm behavior
for module in model.modules():
    if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
        module.eval()
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eval()

# Load face detector
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]

# Emotion colors for bounding boxes
emotion_colors = {
    "Neutral": (255, 255, 255),  # White
    "Happy": (0, 255, 255),      # Yellow
    "Surprise": (0, 165, 255),   # Orange
    "Sad": (255, 0, 0),          # Blue
    "Angry": (0, 0, 255),        # Red
    "Disgust": (128, 0, 128),    # Purple
    "Fear": (255, 255, 0),       # Cyan
    "Contempt": (0, 255, 0)      # Green
}

# Inside Out character mapping
inside_out_characters = {
    "Neutral": "static//Neutral.gif",      # You can use a calm character
    "Happy": "static//Happy.gif",            # Joy
    "Surprise": "static//Surprise.gif",    # Could be Joy or custom
    "Sad": "static//Sad.gif",          # Sadness
    "Angry": "static//Angry.gif",          # Anger
    "Disgust": "static//Disgust.gif",      # Disgust
    "Fear": "static//Fear.gif",            # Fear
    "Contempt": "static//Contempt.gif"     # Could be Disgust or custom
}

class GIFLoader:
    def __init__(self):
        self.gif_frames = {}
        self.gif_frame_counts = {}
        self.current_frame_indices = {}
        self.frame_counter = 0
        self.load_gifs()
    
    def load_gifs(self):
        """Load all GIF files and extract frames"""
        for emotion, gif_path in inside_out_characters.items():
            if os.path.exists(gif_path):
                try:
                    gif = Image.open(gif_path)
                    frames = []
                    
                    for frame in ImageSequence.Iterator(gif):
                        # Convert to RGB and then to BGR for OpenCV
                        frame_rgb = frame.convert('RGBA')
                        frame_array = np.array(frame_rgb)
                        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGRA)
                        frames.append(frame_bgr)
                    
                    self.gif_frames[emotion] = frames
                    self.gif_frame_counts[emotion] = len(frames)
                    self.current_frame_indices[emotion] = 0
                    
                except Exception as e:
                    print(f"Error loading GIF for {emotion}: {e}")
                    self.gif_frames[emotion] = None
            else:
                print(f"GIF file not found: {gif_path}")
                self.gif_frames[emotion] = None
    
    def get_current_frame(self, emotion):
        """Get current frame for an emotion and advance frame index"""
        if emotion not in self.gif_frames or self.gif_frames[emotion] is None:
            return None
            
        frames = self.gif_frames[emotion]
        current_idx = self.current_frame_indices[emotion]
        
        # Advance frame every few video frames to control GIF speed
        if self.frame_counter % 3 == 0:  # Change GIF frame every 3 video frames
            self.current_frame_indices[emotion] = (current_idx + 1) % self.gif_frame_counts[emotion]
        
        return frames[current_idx]
    
    def update_frame_counter(self):
        """Update global frame counter"""
        self.frame_counter += 1

# Emotion text colors (cycling colors for top emotion)
emotion_text_colors = {
    "Neutral": [(255,255,255), (224,212,196), (228,203,179)],
    "Happy": [(182,110,68), (76,235,253), (83,169,242)],
    "Surprise": [(247,255,0), (42,42,165), (232,206,0)],
    "Sad": [(194,105,3), (228,172,32), (237,202,162)],
    "Angry": [(61, 57, 242), (49,121,249), (232,220,214)],
    "Disgust": [(70,190,77), (120,159,6), (100,55,124)],
    "Fear": [(198, 128, 134), (133,71,68), (80,45,98)],
    "Contempt": [(160, 134, 72), (145, 180, 250), (173, 217, 251)]
}

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.frame_count = 0
        self.color_index = 0
        self.faces = []
        self.gif_loader = GIFLoader()
        
    def __del__(self):
        self.video.release()
    
    def overlay_gif_on_frame(self, frame, gif_frame, x, y, width, height):
        """Overlay GIF frame on video frame with transparency support"""
        if gif_frame is None:
            return frame
            
        # Resize GIF frame to desired size
        gif_resized = cv2.resize(gif_frame, (width, height))
        
        # Handle transparency if GIF has alpha channel
        if gif_resized.shape[2] == 4:  # BGRA
            # Extract alpha channel
            alpha = gif_resized[:, :, 3] / 255.0
            
            # Ensure we don't go outside frame boundaries
            y1, y2 = max(0, y), min(frame.shape[0], y + height)
            x1, x2 = max(0, x), min(frame.shape[1], x + width)
            
            # Adjust GIF coordinates if clipped
            gy1, gy2 = max(0, -y), height - max(0, (y + height) - frame.shape[0])
            gx1, gx2 = max(0, -x), width - max(0, (x + width) - frame.shape[1])
            
            if y2 > y1 and x2 > x1 and gy2 > gy1 and gx2 > gx1:
                # Apply alpha blending
                for c in range(3):  # BGR channels
                    frame[y1:y2, x1:x2, c] = (
                        alpha[gy1:gy2, gx1:gx2] * gif_resized[gy1:gy2, gx1:gx2, c] +
                        (1 - alpha[gy1:gy2, gx1:gx2]) * frame[y1:y2, x1:x2, c]
                    )
        else:  # No alpha channel, direct overlay
            y1, y2 = max(0, y), min(frame.shape[0], y + height)
            x1, x2 = max(0, x), min(frame.shape[1], x + width)
            gy1, gy2 = max(0, -y), height - max(0, (y + height) - frame.shape[0])
            gx1, gx2 = max(0, -x), width - max(0, (x + width) - frame.shape[1])
            
            if y2 > y1 and x2 > x1:
                frame[y1:y2, x1:x2] = gif_resized[gy1:gy2, gx1:gx2, :3]
        
        return frame
        
    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Update GIF frame counter
        self.gif_loader.update_frame_counter()
        
        # Change color every 10 frames
        if self.frame_count % 10 == 0:
            self.color_index = (self.color_index + 1) % 3
            
        # Run face detection every 3 frames to reduce lag
        if self.frame_count % 3 == 0:
            self.faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48))
        
        self.frame_count += 1
        
        for (x, y, w, h) in self.faces:
            # Extract and preprocess face
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (48, 48))
            face_img = np.expand_dims(face_img, axis=0)
            face_tensor = torch.tensor(face_img, dtype=torch.float32).div(255).sub(0.5).div(0.5).unsqueeze(0).to(device)
            
            # Run model inference
            with torch.no_grad():
                outputs = model(face_tensor)
                probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                top_emotion_idx = np.argmax(probs)
                top_emotion = emotions[top_emotion_idx]
            
            # Draw bounding box with emotion color
            box_color = emotion_colors.get(top_emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            
            # Add Inside Out character GIF for top emotion
            gif_frame = self.gif_loader.get_current_frame(top_emotion)
            if gif_frame is not None:
                # Position GIF above the face
                gif_size = min(w, h) // 2  # Make GIF half the size of face
                gif_x = x + w // 2 - gif_size // 2  # Center horizontally
                gif_y = y - gif_size - 10  # Position above face
                
                frame = self.overlay_gif_on_frame(frame, gif_frame, gif_x, gif_y, gif_size, gif_size)
            
            # Display emotion probabilities
            text_start_y = y - 20 if gif_frame is None else y - gif_size - 30
            for i, (emotion, prob) in enumerate(zip(emotions, probs)):
                if i == top_emotion_idx:
                    text_color = emotion_text_colors[top_emotion][self.color_index]
                else:
                    text_color = (255, 255, 255)
                    
                text = f"{emotion}: {int(prob * 100)}%"
                cv2.putText(frame, text, (x, text_start_y - (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        return frame

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)