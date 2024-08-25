import cv2
import tensorflow as tf
import ffmpeg
import firebase_admin
from firebase_admin import credentials, storage, firestore
import os

# Initialize Firebase
cred = credentials.Certificate("path/to/firebase-adminsdk.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': '<backend_hackathon>.appspot.com'
})
db = firestore.client()

def upload_video_to_firebase(video_path, video_name):
    """Uploads the video to Firebase storage and stores the link in Firestore."""
    bucket = storage.bucket()
    blob = bucket.blob(f'edited_videos/{video_name}')
    blob.upload_from_filename(video_path)
    blob.make_public()

    # Save the video link in Firestore
    video_data = {
        'name': video_name,
        'url': blob.public_url
    }
    db.collection('videos').add(video_data)
    print(f"Video uploaded to Firebase: {blob.public_url}")

def analyze_video(video_path):
    """Analyzes the video using TensorFlow."""
    # Load a pre-trained model (example: MobileNet)
    model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)
    
    # Process video frames
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = []
    
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame for model input
        frame_resized = cv2.resize(frame, (224, 224))
        frame_array = tf.keras.applications.mobilenet_v2.preprocess_input(frame_resized)
        frame_array = tf.expand_dims(frame_array, 0)

        # Analyze frame
        predictions = model.predict(frame_array)
        processed_frames.append(predictions)  # You can store or use predictions
        
    cap.release()
    return processed_frames

def encode_video_with_ffmpeg(input_video_path, output_video_path):
    """Encodes the video with FFMPEG."""
    (
        ffmpeg
        .input(input_video_path)
        .output(output_video_path, vcodec='libx264', crf=23, preset='medium')
        .run()
    )
    print(f"Video encoded and saved to: {output_video_path}")

def process_video(video_path):
    """Handles the full video processing pipeline."""
    video_name = os.path.basename(video_path)
    
    # Analyze the video
    analyze_video(video_path)

    # Encode the video with FFMPEG
    output_video_path = f"edited_{video_name}"
    encode_video_with_ffmpeg(video_path, output_video_path)

    # Upload the edited video to Firebase
    upload_video_to_firebase(output_video_path, video_name)

    # Clean up local files if needed
    os.remove(output_video_path)

# Example usage:
# process_video("path/to/your/video.mp4")
#4. Integrate the Module into Your Project
#You can now use this module in your main.py or other parts of your project.

#python
#Copy code
# main.py

from src.video_processor import process_video

def main():
    video_path = "path/to/your/video.mp4"
    process_video(video_path)

if __name__ == "__main__":
    main()