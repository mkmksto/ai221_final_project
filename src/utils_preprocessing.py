"""
Utilities for preprocessing the images for better model performance.

Also includes some feature extraction methods.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from .utils_data import RAW_DATA_DF


def detect_edges_and_lines(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect edges and lines in an image using Canny edge detection and Hough transform.

    Args:
        image: Input image as numpy array

    Returns:
        tuple containing:
            - edges: Edge detection result
            - line_image: Original image with detected lines drawn
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image

    # Apply Canny edge detection
    edges = cv2.Canny(img_gray, threshold1=50, threshold2=150)

    # Apply Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # Create a copy of original image to draw lines on
    line_image = image.copy()

    # Draw detected lines
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the edge detection result
    plt.figure()
    plt.imshow(edges, cmap="gray")
    plt.axis("off")
    plt.title("Canny Edge Detection")
    plt.show()

    # Display the line detection result
    plt.figure()
    plt.imshow(line_image)
    plt.axis("off")
    plt.title("Hough Line Transform")
    plt.show()

    return edges, line_image


# Feature Extraction methods
def extract_glcm_features(
    image: np.ndarray,
    distances: list[int] = [1],
    angles: list[float] = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
) -> dict[str, float]:
    """Extract GLCM (Gray-Level Co-occurrence Matrix) features from an image.

    Args:
        image (np.ndarray): Input image
        distances (list[int], optional): List of pixel pair distances. Defaults to [1].
        angles (list[float], optional): List of pixel pair angles in radians.
            Defaults to [0, pi/4, pi/2, 3*pi/4].

    Returns:
        dict[str, float]: Dictionary containing GLCM features:
            - contrast: Measure of local intensity variation
            - dissimilarity: Similar to contrast but increases linearly
            - homogeneity: Closeness of elements distribution
            - energy: Sum of squared elements (textural uniformity)
            - correlation: Linear dependency of gray levels
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image to reduce number of intensity levels
    bins = 8
    image = (image / 256 * bins).astype(np.uint8)

    # Calculate GLCM properties for each distance/angle
    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=bins,
        symmetric=True,
        normed=True,
    )

    # Extract features
    features = {
        "contrast": graycoprops(glcm, "contrast").mean(),
        "dissimilarity": graycoprops(glcm, "dissimilarity").mean(),
        "homogeneity": graycoprops(glcm, "homogeneity").mean(),
        "energy": graycoprops(glcm, "energy").mean(),
        "correlation": graycoprops(glcm, "correlation").mean(),
    }

    return features


def visualize_glcm(image: np.ndarray, distance: int = 1, angle: float = 0) -> None:
    """Visualize GLCM matrix for an image.

    Args:
        image (np.ndarray): Input image
        distance (int, optional): Pixel pair distance. Defaults to 1.
        angle (float, optional): Pixel pair angle in radians. Defaults to 0.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image to reduce number of intensity levels
    bins = 8
    image = (image / 256 * bins).astype(np.uint8)

    # Calculate GLCM
    glcm = graycomatrix(
        image,
        distances=[distance],
        angles=[angle],
        levels=bins,
        symmetric=True,
        normed=True,
    )

    # Plot original image and GLCM
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(image, cmap="gray")
    ax1.set_title("Original Image")
    ax1.axis("off")

    im = ax2.imshow(glcm[:, :, 0, 0], cmap="viridis")
    ax2.set_title(f"GLCM (d={distance}, θ={angle:.2f})")
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    plt.show()


def preprocess_leaf_image(image: np.ndarray) -> np.ndarray:
    """Preprocess leaf image by:
    1. Converting to appropriate color space
    2. Removing background/segmenting leaf
    3. Reducing noise
    4. Enhancing contrast
    5. Normalizing

    Args:
        image: Input image as numpy array

    Returns:
        Preprocessed image as numpy array
    """
    # 1. Convert to LAB color space for better segmentation
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # 2. Segment leaf using Otsu's thresholding on A channel
    _, mask = cv2.threshold(
        lab_image[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    mask = mask.astype(np.uint8)

    # Clean up mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply mask to original image
    segmented = cv2.bitwise_and(image, image, mask=mask)

    # 3. Reduce noise with bilateral filter
    denoised = cv2.bilateralFilter(segmented, d=9, sigmaColor=75, sigmaSpace=75)

    # 4. Enhance contrast with CLAHE
    lab_enhanced = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_enhanced[:, :, 0] = clahe.apply(lab_enhanced[:, :, 0])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 5. Normalize to [0,1] range
    normalized = enhanced.astype(np.float32) / 255.0

    return normalized


def extract_shape_features(image: np.ndarray) -> dict[str, float]:
    """Extract shape features from preprocessed leaf image:
    - Area
    - Perimeter
    - Circularity
    - Eccentricity
    - Major/minor axis lengths
    - Aspect ratio
    - Form factor
    - Rectangularity
    - Narrow factor
    """
    pass


def extract_texture_features(image: np.ndarray) -> dict[str, float]:
    """Extract texture features from preprocessed leaf image:
    - GLCM features (already implemented above)
    - Local Binary Patterns
    - Gabor filter responses
    - Edge density
    - Roughness metrics
    """
    pass


def extract_color_features(image: np.ndarray) -> dict[str, float]:
    """Extract color features from original leaf image:
    - Color moments (mean, std, skewness)
    - Color histograms
    - Dominant colors
    - Color ratios
    For multiple color spaces (RGB, HSV, Lab)
    """
    pass


def extract_vein_features(image: np.ndarray) -> dict[str, float]:
    """Extract vein features from preprocessed leaf image:
    - Vein density
    - Vein orientation histogram
    - Vein branching points
    - Vein length statistics
    """
    pass


def extract_all_features(image_path: str) -> dict[str, float]:
    """Extract all features from a leaf image by:
    1. Loading image
    2. Preprocessing
    3. Extracting all feature types
    4. Combining into single feature vector
    """
    pass


def create_feature_dataset(data_df: pd.DataFrame) -> pd.DataFrame:
    """Create dataset with extracted features for all images:
    1. Iterate through all image files
    2. Extract features for each
    3. Create DataFrame with features and labels
    Returns DataFrame with columns:
    - image_filename
    - all extracted features
    - class_number (label)
    """
    pass


if __name__ == "__main__":

    print("Hello World from `utils_preprocessing.py`")
    print(RAW_DATA_DF.head(5))
