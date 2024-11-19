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


def create_bg_rem_mask(image: np.ndarray) -> np.ndarray:
    """Create a mask for background removal from leaf image using rembg.

    Args:
        image: Input image as numpy array (RGB format)

    Returns:
        Mask array as numpy array where 0 represents background and 255 represents foreground
    """
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image)

    # Remove background using rembg
    output = remove(pil_image)

    # Convert to numpy array and extract alpha channel as mask
    output_array = np.array(output)
    mask = output_array[:, :, 3]  # Alpha channel

    # Normalize mask to 0-255 range
    mask = ((mask > 0) * 255).astype(np.uint8)

    return mask


def remove_background(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove background from image using a binary mask.

    Args:
        image: Input image as numpy array (RGB format)
        mask: Binary mask where 0 represents background and 255 represents foreground

    Returns:
        Image with background removed (RGB format with transparent background)
    """
    # Ensure mask is binary
    binary_mask = mask > 128

    # Create output image with alpha channel
    output = np.zeros((*image.shape[:2], 4), dtype=np.uint8)

    # Copy RGB channels
    output[:, :, :3] = image

    # Set alpha channel from mask
    output[:, :, 3] = binary_mask * 255

    return output


def extract_shape_features(image: np.ndarray) -> dict[str, float]:
    """Extract shape features from preprocessed leaf image with removed background.

    Args:
        image: Input image as numpy array with removed background (RGBA format)

    Returns:
        dict[str, float]: Dictionary containing shape features
    """
    # Use alpha channel as mask
    binary_mask = (image[:, :, 3] > 128).astype(np.uint8) * 255

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


def extract_all_features(
    original_image_path: Path, bg_removed_image_path: Path
) -> dict[str, float]:
    """Extract all features from a leaf image.

    Args:
        original_image_path: Path
            Path to the original image file
        bg_removed_image_path: Path
            Path to the image file with background removed

    Returns:
        dict[str, float]: Combined dictionary containing all features
    """
    # Load images
    original_image = plt.imread(original_image_path)
    bg_removed_image = plt.imread(bg_removed_image_path)

    # Convert to uint8 if necessary
    if original_image.dtype == np.float32 or original_image.dtype == np.float64:
        original_image = (original_image * 255).astype(np.uint8)
    if bg_removed_image.dtype == np.float32 or bg_removed_image.dtype == np.float64:
        bg_removed_image = (bg_removed_image * 255).astype(np.uint8)

    # Extract features using appropriate image versions
    # Use original image for color features and vein features
    color_features = extract_color_features(original_image)
    vein_features = extract_vein_features(original_image)

    # Use background-removed image for texture and shape features
    texture_features = extract_texture_features(bg_removed_image)
    shape_features = extract_shape_features(bg_removed_image)

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
            DataFrame containing folder paths and labels
            Must have columns: ['original_image_path', 'bg_removed_image_path',
                              'class_name', 'class_number']
            where *_image_path columns contain paths to folders with .webp images

    Returns:
        pd.DataFrame: DataFrame containing:
            - image_filename: str (basename of the image file)
            - class_name: str (name of the class)
            - class_number: int (numeric label)
            - feature columns: float (all extracted features)

    Raises:
        ValueError: If required columns are missing from data_df
    """
    # Verify required columns exist
    required_cols = [
        "original_image_path",
        "bg_removed_image_path",
        "class_name",
        "class_number",
    ]
    if not all(col in data_df.columns for col in required_cols):
        raise ValueError(f"data_df must contain columns: {required_cols}")

    # Initialize lists to store results
    all_features: list[Dict[str, Union[str, float, int]]] = []

    # Process each class folder
    for idx, row in tqdm(
        data_df.iterrows(), total=len(data_df), desc="Processing classes"
    ):
        print(f"processing folder class: {row['class_name']}")
        # # do not delete this
        # if idx >= 3:
        #     break

        original_folder = Path(row["original_image_path"])
        bg_removed_folder = Path(row["bg_removed_image_path"])

        # Get all .webp files in original folder
        original_images = list(original_folder.glob("*.webp"))

        # Process each image in the class
        for orig_img_path in original_images:
            try:
                # Construct path to corresponding bg-removed image
                bg_removed_img_path = bg_removed_folder / orig_img_path.name

                if not bg_removed_img_path.exists():
                    print(
                        f"Warning: No matching bg-removed image for "
                        f"{orig_img_path.name}"
                    )
                    continue

                # Extract features using both image paths
                features = extract_all_features(orig_img_path, bg_removed_img_path)

                # Add metadata
                features["image_filename"] = orig_img_path.name
                features["class_name"] = row["class_name"]
                features["class_number"] = row["class_number"]

                all_features.append(features)

            except Exception as e:
                print(f"Error processing {orig_img_path}: {str(e)}")
                print(f"Error traceback: {str(e)}")
                continue

    # Create DataFrame from all features
    feature_df = pd.DataFrame(all_features)

    # Ensure consistent column ordering
    # Move metadata columns to front
    metadata_cols = ["image_filename", "class_name", "class_number"]
    feature_cols = [col for col in feature_df.columns if col not in metadata_cols]
    feature_df = feature_df[metadata_cols + feature_cols]

    return feature_df


if __name__ == "__main__":

    print("Hello World from `utils_preprocessing.py`")
    print(RAW_DATA_DF.head(5))
