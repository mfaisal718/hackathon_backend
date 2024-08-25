# main.py
import numpy as np
import tensorflow as tf
from src.my_module import square_tensor
from src.video_processor import process_video
from src.face_emoji_processor import process_video_with_emoji

def main():
    video_path = "path/to/your/video.mp4"
    output_video_path = "output_with_emoji.mp4"
    emoji_path = "path/to/emoji.png"

    process_video_with_emoji(video_path, output_video_path, emoji_path)

    # Example: Simple NumPy array and TensorFlow operation
    numpy_array = np.array([1, 2, 3, 4, 5])
    print(f"NumPy Array: {numpy_array}")


    # Simple TensorFlow operation
    tensor = tf.constant(numpy_array)
    squared_tensor = tf.square(tensor)
    print(f"Squared TensorFlow Tensor: {squared_tensor.numpy()}")
    video_path = "path/to/your/video.mp4"
    process_video(video_path)

if __name__ == "__main__":
    main()