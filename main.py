# main.py
import numpy as np
import tensorflow as tf
from src.my_module import square_tensor
from src.video_processor import process_video

def main():
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