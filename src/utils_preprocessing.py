"""
Utilities for preprocessing the images for better model performance.

Also includes some feature extraction methods.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    """Preprocess leaf image with basic enhancements.

    Args:
        image: Input image as numpy array

    Returns:
        Preprocessed image as numpy array
    """

    # Convert BGR to RGB if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply contrast enhancement using CLAHE
    if len(image.shape) == 3:
        # Apply CLAHE to L channel in LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # Apply CLAHE directly to grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

    # Reduce noise while preserving edges
    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Normalize pixel values to [0,255] range
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return image


def extract_shape_features(image: np.ndarray) -> dict[str, float]:
    """Extract shape features from preprocessed leaf image.

    Args:
        image: Input image as numpy array, either RGB or grayscale

    Returns:
        Dictionary containing the following shape features:
            - area: Area of the leaf contour
            - perimeter: Perimeter length of the leaf contour
            - circularity: How circular the leaf shape is (4π*area/perimeter²)
            - eccentricity: Eccentricity of fitted ellipse (0=circle, 1=line)
            - major_axis_length: Length of major axis of fitted ellipse
            - minor_axis_length: Length of minor axis of fitted ellipse
            - aspect_ratio: Ratio of major to minor axis lengths
            - form_factor: Another circularity measure (4π*area/perimeter²)
            - rectangularity: How rectangular the leaf is (area/bounding_box_area)
            - narrow_factor: Width to height ratio of bounding box
    """
    # Convert to binary mask if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}

    # Get largest contour
    contour = max(contours, key=cv2.contourArea)

    # Calculate basic measurements
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Fit ellipse to get major/minor axes
    if len(contour) >= 5:  # Need at least 5 points to fit ellipse
        (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
    else:
        major_axis = minor_axis = 1.0

    # Calculate shape features
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    eccentricity = (
        np.sqrt(1 - (minor_axis / major_axis) ** 2)
        if major_axis > 0 and minor_axis <= major_axis
        else 0
    )
    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
    form_factor = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    rectangularity = area / (w * h) if (w * h) > 0 else 0

    # Narrow factor
    narrow_factor = w / h if h > 0 else 0

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "circularity": float(circularity),
        "eccentricity": float(eccentricity),
        "major_axis_length": float(major_axis),
        "minor_axis_length": float(minor_axis),
        "aspect_ratio": float(aspect_ratio),
        "form_factor": float(form_factor),
        "rectangularity": float(rectangularity),
        "narrow_factor": float(narrow_factor),
    }


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
