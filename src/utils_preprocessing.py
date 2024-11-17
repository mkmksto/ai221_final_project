"""
Utilities for preprocessing the images for better model performance.

Also includes some feature extraction methods.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops


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
