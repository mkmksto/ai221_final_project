"""Basic Plotting Utilities"""

import matplotlib.pyplot as plt
import numpy as np


def plot_5x8_grid(processed_images: list[np.ndarray]) -> None:
    """Plot preprocessed images in a 5x8 grid layout.

    Args:
        processed_images: List of preprocessed images as numpy arrays
    """
    # Create a 5x8 grid of subplots
    fig, axes = plt.subplots(5, 8, figsize=(15, 8))

    # Plot each preprocessed image
    for idx, img in enumerate(processed_images):
        row = idx // 8
        col = idx % 8

        # Display image
        axes[row, col].imshow(img)
        axes[row, col].axis("off")

        # Add index as title
        axes[row, col].set_title(f"Image {idx+1}", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_images_side_by_side(
    image1: np.ndarray,
    image2: np.ndarray,
    title1: str = "Image 1",
    title2: str = "Image 2",
) -> None:
    """Plot two images side by side for comparison.

    Args:
        image1: First image as numpy array
        image2: Second image as numpy array
        title1: Title for first image. Defaults to "Image 1"
        title2: Title for second image. Defaults to "Image 2"
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(image1)
    ax1.set_title(title1)
    ax1.axis("off")

    ax2.imshow(image2)
    ax2.set_title(title2)
    ax2.axis("off")

    plt.tight_layout()
    plt.show()
