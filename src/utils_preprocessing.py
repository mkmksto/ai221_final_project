"""
Utilities for preprocessing the images for better model performance.

Also includes some feature extraction methods.
"""

import os
from pathlib import Path
from typing import Dict, Union

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from PIL import Image
from rembg import remove
from scipy.ndimage import label
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import frangi, gabor
from skimage.morphology import skeletonize, thin
from sklearn.cluster import KMeans
from tqdm import tqdm

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


def create_bg_rem_mask(pil_image: Image.Image) -> np.ndarray:
    """Create a mask for background removal from leaf image using rembg.

    Args:
        pil_image: Input PIL image

    Returns:
        Mask array as numpy array where 0 represents background and 255 represents foreground
    """
    # Remove background using rembg
    output = remove(pil_image)

    # Convert to numpy array and extract alpha channel as mask
    output_array = np.array(output)
    mask = output_array[:, :, 3]  # Alpha channel

    # Normalize mask to 0-255 range
    mask = ((mask > 0) * 255).astype(np.uint8)

    return mask

    # pipe = pipeline(
    #     "image-segmentation",
    #     model="briaai/RMBG-1.4",
    #     trust_remote_code=True,
    #     device="cuda",
    # )

    # # Get mask from transformer
    # mask = pipe(pil_image, return_mask=True)
    # mask_array = np.array(mask)

    # return mask_array


def extract_shape_features(image: np.ndarray) -> dict[str, float]:
    """Extract shape features from preprocessed leaf
    use this for background removal: https://huggingface.co/briaai/RMBG-1.4
    Loading the model works like this:
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    pillow_mask = pipe(image_path, return_mask = True) # outputs a pillow mask
    pillow_image = pipe(image_path)
    only use what's necessary from the transformer pipeline

    Args:
        image: np.ndarray
            Input image as numpy array, either RGB or grayscale

    Returns:
        dict[str, float]: Dictionary containing shape features:
            - area: Total leaf area in pixels
            - perimeter: Length of leaf boundary in pixels
            - circularity: How circular the leaf shape is (1=perfect circle)
            - eccentricity: How elongated the leaf is
            - solidity: Ratio of leaf area to convex hull area
            - extent: Ratio of leaf area to bounding rect area
    """
    # Convert image to PIL format for transformer
    pil_image = Image.fromarray(image)

    # Initialize background removal pipeline
    mask_array = create_bg_rem_mask(pil_image)

    # Convert mask to binary
    binary_mask = (mask_array > 128).astype(np.uint8) * 255
    # Visualize binary mask
    plt.figure(figsize=(8, 8))
    plt.imshow(binary_mask, cmap="gray")
    plt.title("Binary Mask")
    plt.axis("off")
    plt.show()

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return {}

    # Get largest contour (main leaf)
    contour = max(contours, key=cv2.contourArea)

    # Calculate features
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Circularity = 4*pi*area/perimeter^2 (1 for perfect circle)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

    # # Fit ellipse for eccentricity
    # if len(contour) >= 5:
    #     (_, _), (major, minor), _ = cv2.fitEllipse(contour)
    #     eccentricity = np.sqrt(1 - (minor / major) ** 2) if major > 0 else 0
    # else:
    #     eccentricity = 0

    # Calculate eccentricity using moments
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        mu20 = moments["mu20"] / moments["m00"]
        mu02 = moments["mu02"] / moments["m00"]
        mu11 = moments["mu11"] / moments["m00"]
        temp = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11**2)
        eccentricity = float(np.sqrt(2 * temp / (mu20 + mu02 + temp)))
    else:
        eccentricity = 1.0

    # Calculate solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # Calculate extent
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "circularity": float(circularity),
        "eccentricity": float(eccentricity),
        "solidity": float(solidity),
        "extent": float(extent),
    }


def extract_texture_features(image: np.ndarray) -> dict[str, float]:
    """Extract texture features from preprocessed leaf image.

    Args:
        image: Input image as numpy array, either RGB or grayscale

    Returns:
        Dictionary containing texture features:
            - GLCM features (contrast, dissimilarity, homogeneity, energy, correlation)
            - LBP histogram features
            - Edge density
            - Roughness metrics based on gradient magnitudes
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = gray = image.copy()

    # Calculate GLCM features
    distances = [1, 3, 5]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(
        gray, distances=distances, angles=angles, symmetric=True, normed=True
    )

    glcm_features = {}
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    for prop in props:
        feature = graycoprops(glcm, prop).mean()
        glcm_features[f"glcm_{prop}"] = float(feature)

    # Calculate LBP features
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
    )
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7

    lbp_features = {f"lbp_hist_{i}": float(v) for i, v in enumerate(hist)}

    # Calculate edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0) / edges.size)

    # Calculate roughness metrics using gradient magnitudes
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    roughness = float(np.mean(gradient_magnitude))
    roughness_std = float(np.std(gradient_magnitude))

    # Combine all features
    features = {
        **glcm_features,
        **lbp_features,
        "edge_density": edge_density,
        "roughness_mean": roughness,
        "roughness_std": roughness_std,
    }

    return features


def extract_color_features(image: np.ndarray) -> dict[str, float]:
    """Extract color features from original leaf image.

    Args:
        image: np.ndarray
            Input image in BGR format (OpenCV default)

    Returns:
        dict[str, float]: Dictionary containing color features:
            - Color moments (mean, std, skewness) for each channel in RGB, HSV, Lab
            - Color histogram features for each channel
            - Dominant color ratios
            - Color space ratios
    """
    # Convert to different color spaces
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    features = {}

    # Extract color moments for each color space
    for color_space, img in [("rgb", rgb_img), ("hsv", hsv_img), ("lab", lab_img)]:
        for i, channel in enumerate(cv2.split(img)):
            mean = float(np.mean(channel))
            std = float(np.std(channel))
            skewness = float(scipy.stats.skew(channel.ravel()))

            features[f"{color_space}_{i}_mean"] = mean
            features[f"{color_space}_{i}_std"] = std
            features[f"{color_space}_{i}_skewness"] = skewness

    # Calculate color histograms
    for color_space, img in [("rgb", rgb_img), ("hsv", hsv_img), ("lab", lab_img)]:
        for i, channel in enumerate(cv2.split(img)):
            hist = cv2.calcHist([channel], [0], None, [32], [0, 256])
            hist = hist.flatten() / hist.sum()

            for j, val in enumerate(hist):
                features[f"{color_space}_{i}_hist_{j}"] = float(val)

    # Calculate dominant colors using k-means
    pixels = rgb_img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5, random_state=42).fit(pixels)
    colors = kmeans.cluster_centers_
    color_labels = kmeans.labels_

    # Calculate ratios of dominant colors
    unique_labels, counts = np.unique(color_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        features[f"dominant_color_{label}_ratio"] = float(count / len(color_labels))

    # Calculate color space ratios
    features["green_ratio"] = float(
        np.mean(rgb_img[:, :, 1])
        / (np.mean(rgb_img[:, :, 0]) + np.mean(rgb_img[:, :, 2]) + 1e-7)
    )
    features["saturation_ratio"] = float(np.mean(hsv_img[:, :, 1]) / 255.0)
    features["value_ratio"] = float(np.mean(hsv_img[:, :, 2]) / 255.0)

    return features


def extract_vein_features(image: np.ndarray) -> dict[str, float]:
    """Extract vein features from preprocessed leaf image.

    This function analyzes the vein structure of a leaf using various image processing
    techniques including Frangi vesselness filter and skeletonization.

    Args:
        image: np.ndarray
            Input image as numpy array, either RGB or grayscale

    Returns:
        dict[str, float]: Dictionary containing vein features:
            - vein_density: Ratio of vein pixels to total pixels
            - mean_orientation: Mean orientation of vein segments
            - orientation_std: Standard deviation of vein orientations
            - branching_density: Density of branching points
            - mean_segment_length: Average length of vein segments
            - segment_length_std: Standard deviation of segment lengths
            - vesselness_mean: Mean vesselness response
            - vesselness_std: Standard deviation of vesselness response
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply Frangi vesselness filter to enhance veins
    vesselness = frangi(
        gray, scale_range=(1, 5), scale_step=1, beta=0.5, black_ridges=False
    )

    # Normalize vesselness response
    vesselness = (vesselness - vesselness.min()) / (
        vesselness.max() - vesselness.min() + 1e-8
    )

    # Threshold vesselness to get binary vein mask
    vein_mask = vesselness > 0.1

    # Skeletonize the vein mask
    skeleton = skeletonize(vein_mask)

    # Calculate vein density
    vein_density = float(np.sum(skeleton) / skeleton.size)

    # Calculate vesselness statistics
    vesselness_mean = float(np.mean(vesselness))
    vesselness_std = float(np.std(vesselness))

    # Find branching points using connected components
    # Get points with more than 2 neighbors
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton.astype(np.float32), -1, kernel)
    branch_points = (neighbors > 3) & skeleton
    branching_density = float(np.sum(branch_points) / skeleton.size)

    # Calculate segment orientations using Gabor filter bank
    orientations = []
    for theta in np.linspace(0, np.pi, 8):
        real, _ = gabor(gray, frequency=0.6, theta=theta)
        response = np.sum(real * skeleton)
        orientations.append((theta, response))

    # Get orientation statistics
    orientation_responses = np.array([resp for _, resp in orientations])
    weighted_orientations = np.array([theta for theta, _ in orientations])
    mean_orientation = float(
        np.average(weighted_orientations, weights=orientation_responses)
    )
    orientation_std = float(np.std(weighted_orientations))

    # Calculate segment lengths
    # Label individual segments
    labeled_skeleton, num_segments = label(skeleton)
    segment_lengths = []

    for i in range(1, num_segments + 1):
        segment = labeled_skeleton == i
        segment_lengths.append(np.sum(segment))

    if segment_lengths:
        mean_segment_length = float(np.mean(segment_lengths))
        segment_length_std = float(np.std(segment_lengths))
    else:
        mean_segment_length = 0.0
        segment_length_std = 0.0

    return {
        "vein_density": vein_density,
        "mean_orientation": mean_orientation,
        "orientation_std": orientation_std,
        "branching_density": branching_density,
        "mean_segment_length": mean_segment_length,
        "segment_length_std": segment_length_std,
        "vesselness_mean": vesselness_mean,
        "vesselness_std": vesselness_std,
    }


# def extract_all_features(image_path: str) -> dict[str, float]:
def extract_all_features(image_path: Path) -> dict[str, float]:
    """Extract all features from a leaf image.

    This function loads an image, preprocesses it, and extracts all available features
    including color, texture, shape, and vein features.

    Args:
        image_path: str
            Path to the image file

    Returns:
        dict[str, float]: Combined dictionary containing all features:
            - Color features (color moments, histograms, dominant colors)
            - Texture features (GLCM, LBP, edge density, roughness)
            - Shape features (area, perimeter, form factor, etc.)
            - Vein features (density, orientation, branching points)
    """
    # Read image
    image = cv2.imread(image_path.as_posix())
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Preprocess image
    preprocessed_image = preprocess_leaf_image(image)

    # Extract all feature types
    color_features = extract_color_features(image)  # Use original image for color
    texture_features = extract_texture_features(preprocessed_image)
    shape_features = extract_shape_features(
        preprocessed_image
    )  # TODO: verify this actually works
    vein_features = extract_vein_features(preprocessed_image)

    # Combine all features
    all_features = {
        **color_features,
        **texture_features,
        **shape_features,
        **vein_features,
    }

    return all_features


def create_feature_dataset(data_df: pd.DataFrame) -> pd.DataFrame:
    """Create dataset with extracted features for all images.

    This function processes all images in the dataset, extracts features,
    and creates a DataFrame containing all features and labels.

    Args:
        data_df: pd.DataFrame
            DataFrame containing image paths and labels
            Must have columns: ['image_path', 'class_number']

    Returns:
        pd.DataFrame: DataFrame containing:
            - image_filename: str (basename of the image file)
            - feature columns: float (all extracted features)
            - class_number: int (label)

    Raises:
        ValueError: If required columns are missing from data_df
    """
    # Verify required columns exist
    required_cols = ["image_path", "class_number"]
    if not all(col in data_df.columns for col in required_cols):
        raise ValueError(f"data_df must contain columns: {required_cols}")

    # Initialize lists to store results
    all_features: list[Dict[str, Union[str, float, int]]] = []

    # Process each image with progress bar
    for _, row in tqdm(
        data_df.iterrows(), total=len(data_df), desc="Extracting features"
    ):
        try:
            # Extract features
            features = extract_all_features(row["image_path"])

            # Add image filename and class
            features["image_filename"] = os.path.basename(row["image_path"])
            features["class_number"] = row["class_number"]

            all_features.append(features)

        except Exception as e:
            print(f"Error processing {row['image_path']}: {str(e)}")
            continue

    # Create DataFrame from all features
    feature_df = pd.DataFrame(all_features)

    # Ensure consistent column ordering
    # Move image_filename and class_number to front
    cols = ["image_filename", "class_number"] + [
        col
        for col in feature_df.columns
        if col not in ["image_filename", "class_number"]
    ]
    feature_df = feature_df[cols]

    return feature_df


# -----------------------------------------------------------------------------
# Old functions
# -----------------------------------------------------------------------------


def extract_shape_features_old(image: np.ndarray) -> dict[str, float]:
    """Extract shape features from preprocessed leaf image.

    Args:
        image: Input image as numpy array, either RGB or grayscale

    Returns:
        Dictionary containing shape features
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Create mask to isolate leaf from background
    # Use Otsu's method on blurred image for robust thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up mask with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Invert mask if background is white
    if np.mean(gray[mask == 255]) > np.mean(gray[mask == 0]):
        mask = cv2.bitwise_not(mask)

    # Find contours on the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original image
    axes[0, 0].imshow(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else gray,
        cmap="gray",
    )
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Grayscale
    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title("Grayscale")
    axes[0, 1].axis("off")

    # Binary mask
    axes[1, 0].imshow(mask, cmap="gray")
    axes[1, 0].set_title("Binary Mask")
    axes[1, 0].axis("off")

    # Contour overlay
    result = (
        image.copy()
        if len(image.shape) == 3
        else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    )
    if contours:
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    axes[1, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Detected Contours")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

    # Get largest contour (the leaf)
    cnt = max(contours, key=cv2.contourArea)

    # Create a mask with just the leaf
    leaf_mask = np.zeros_like(mask)
    cv2.drawContours(leaf_mask, [cnt], -1, 255, -1)

    # Basic measurements on the leaf contour
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    # Bounding rectangle features
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 1.0

    # Minimum area rectangle for better extent calculation
    rect = cv2.minAreaRect(cnt)
    rect_area = rect[1][0] * rect[1][1]
    extent = float(area) / rect_area if rect_area > 0 else 1.0

    # Convex hull features
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 1.0

    # Form factor (circularity measure)
    form_factor = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 1.0

    # Calculate compactness
    compactness = float(perimeter * perimeter / (4 * np.pi * area)) if area > 0 else 1.0

    # Calculate eccentricity using moments
    moments = cv2.moments(cnt)
    if moments["m00"] != 0:
        mu20 = moments["mu20"] / moments["m00"]
        mu02 = moments["mu02"] / moments["m00"]
        mu11 = moments["mu11"] / moments["m00"]
        temp = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11**2)
        eccentricity = float(np.sqrt(2 * temp / (mu20 + mu02 + temp)))
    else:
        eccentricity = 1.0

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "form_factor": float(form_factor),
        "aspect_ratio": float(aspect_ratio),
        "extent": float(extent),
        "solidity": float(solidity),
        "compactness": float(compactness),
        "eccentricity": float(eccentricity),
    }


if __name__ == "__main__":

    print("Hello World from `utils_preprocessing.py`")
    print(RAW_DATA_DF.head(5))
