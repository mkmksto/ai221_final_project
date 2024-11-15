"""Utilities for reducing the sizes of the images
into more manageable sizes.

This also includes conversion to the webp for even more compression.
"""

import shutil
from pathlib import Path
from typing import Union

from PIL import Image

DATA_FOLDER = Path(__file__).parent.parent / "data"
RAW_DATA_FOLDER = DATA_FOLDER / "Philippine Medicinal Plant Leaf Dataset(raw)"
PROCESSED_DATA_FOLDER = DATA_FOLDER / "ph_med_plants_reduced_sizes"

# Sample class / folder for testing
HIBISCUS_FOLDER = RAW_DATA_FOLDER / "1Hibiscus rosa-sinensis(HRS)"
CARMONA_RETUSA_FOLDER = RAW_DATA_FOLDER / "16Carmona retusa(CR)"


def process_image_folder(
    input_folder: Union[str, Path], output_folder: Union[str, Path]
) -> None:
    """Process all images in input folder - resize to 500x500 and convert to webp format.

    Args:
        input_folder: Path to folder containing images to process
        output_folder: Path to save processed images, preserving folder structure
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Valid image extensions
    valid_extensions = {".jpg", ".jpeg", ".webp"}

    # Process all files in input folder
    for file_path in input_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            # Create corresponding output path
            rel_path = file_path.relative_to(input_path)
            output_file = output_path / rel_path.parent / f"{rel_path.stem}.webp"

            # Create output subdirectories if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Open and process image
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize to 500x500 maintaining aspect ratio
                img.thumbnail((500, 500))

                # Create new image with exact 500x500 dimensions
                new_img = Image.new("RGB", (500, 500), (255, 255, 255))
                paste_x = (500 - img.width) // 2
                paste_y = (500 - img.height) // 2
                new_img.paste(img, (paste_x, paste_y))

                # Save as webp
                new_img.save(output_file, "webp", quality=80)


if __name__ == "__main__":
    print("Hello World from `utils_image_conversion.py`")
    # print(f"RAW_DATA_FOLDER: {RAW_DATA_FOLDER}")
    # print(f"PROCESSED_DATA_FOLDER: {PROCESSED_DATA_FOLDER}")

    # print(f"HIBISCUS_FOLDER: {HIBISCUS_FOLDER}")
    # process_image_folder(CARMONA_RETUSA_FOLDER, PROCESSED_DATA_FOLDER)
    process_image_folder(RAW_DATA_FOLDER, PROCESSED_DATA_FOLDER)
