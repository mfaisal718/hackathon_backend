from flask import Flask, request, jsonify
import cv2
import tensorflow as tf
import numpy as np
import ffmpeg
import os
import firebase_admin
from firebase_admin import credentials, storage, firestore

app = Flask(__name__)

# Load pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to store uploaded videos
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('emoji-face-1827e-firebase-adminsdk-g4u0m-d4bb22e224.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'emoji-face-1827e.appspot.com'
})
db = firestore.client()

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    # Process the video
    output_path = process_video(filepath)
    
    # Upload the processed video to Firebase Storage
    video_url = upload_to_firebase(output_path)
    
    # Save the video metadata to Firestore
    save_metadata_to_firestore(file.filename, video_url)
    
    return jsonify({"status": "success", "output_url": video_url})

def process_video(video_path):
    # Capture video
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Output video settings
    output_path = os.path.join(UPLOAD_FOLDER, 'output_' + os.path.basename(video_path))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    
    # Load emoji image
    emoji = cv2.imread('emoji.png')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # Resize the emoji to fit the face size
            emoji_resized = cv2.resize(emoji, (w, h))
            
            # Replace face with emoji
            frame[y:y+h, x:x+w] = emoji_resized
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Encode the video using FFMPEG
    encode_video(output_path)
    
    return output_path

def encode_video(video_path):
    output_encoded_path = video_path.replace('.avi', '_encoded.mp4')
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.output(stream, output_encoded_path)
    ffmpeg.run(stream)
    
    return output_encoded_path

def upload_to_firebase(local_file_path):
    # Reference to Firebase storage
    bucket = storage.bucket()
    blob = bucket.blob("videos/{}".format(os.path.basename(local_file_path)))
    
    # Upload the video file
    blob.upload_from_filename(local_file_path)
    
    # Make the file publicly accessible
    blob.make_public()
    
    return blob.public_url
def save_metadata_to_firestore(filename, video_url):
    # Store video metadata in Firestore
    doc_ref = db.collection('videos').document(filename)
    doc_ref.set({
        'filename': filename,
        'video_url': video_url,
        'processed': True
    })

if __name__ == '__main__':
    app.run(debug=True)
