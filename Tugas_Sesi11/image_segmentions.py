import numpy as np
import matplotlib.pyplot as plt
from imageio.v3 import imread
from skimage.color import rgb2gray
from scipy.ndimage import sobel

def load_image(filepath):
    image = imread(filepath)
    if len(image.shape) == 3:  # Convert to grayscale if RGB
        image = rgb2gray(image)
    return image

def sobel_edge_detection(image):
    sobel_x = sobel(image, axis=0)  
    sobel_y = sobel(image, axis=1)  
    sobel_magnitude = np.hypot(sobel_x, sobel_y)  
    return sobel_magnitude / np.max(sobel_magnitude)  

def basic_thresholding(image, threshold):
    binary_image = (image > threshold).astype(np.uint8)
    return binary_image

def main():
    filepath = r"C:\\Users\\deans\\Downloads\\th.jpg"
    try:
        image = load_image(filepath)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    edge_image = sobel_edge_detection(image)
    threshold = 0.5  
    segmented_image = basic_thresholding(edge_image, threshold)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Sobel Edge Detection")
    plt.imshow(edge_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Segmented Image")
    plt.imshow(segmented_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
