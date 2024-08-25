# src/face_emoji_processor.py

import cv2
import ffmpeg
import numpy as np

def load_emoji(emoji_path):
    """Loads the emoji image."""
    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    return emoji

def detect_faces_in_frame(frame, face_cascade):
    """Detects faces in a given frame using OpenCV's Haar Cascade."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def overlay_emoji_on_face(frame, face_coords, emoji):
    """Overlays the emoji on the detected face."""
    for (x, y, w, h) in face_coords:
        # Resize emoji to fit the face
        emoji_resized = cv2.resize(emoji, (w, h))

        # If the emoji has an alpha channel (transparency), we handle blending
        if emoji_resized.shape[2] == 4:
            alpha_s = emoji_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                frame[y:y+h, x:x+w, c] = (alpha_s * emoji_resized[:, :, c] +
                                          alpha_l * frame[y:y+h, x:x+w, c])
        else:
            frame[y:y+h, x:x+w] = emoji_resized

def process_video_with_emoji(video_path, output_video_path, emoji_path):
    """Processes the video to detect faces and overlay emojis."""
    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the emoji image
    emoji = load_emoji(emoji_path)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        faces = detect_faces_in_frame(frame, face_cascade)

        # Overlay emoji on each detected face
        overlay_emoji_on_face(frame, faces, emoji)

        # Write the frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video processed and saved to: {output_video_path}")

# Example usage:
# process_video_with_emoji("path/to/your/video.mp4", "output_with_emoji.mp4", "path/to/emoji.png")