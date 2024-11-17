"""Utilities for reducing the sizes of the images
into more manageable sizes.

This also includes conversion to the webp for even more compression.
"""

from pathlib import Path
from typing import Union

from PIL import Image

from utils_data import PROCESSED_DATA_FOLDER, RAW_DATA_FOLDER

# Sample class / folder for testing


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
    valid_extensions = {".jpg", ".jpeg", ".webp", ".png"}

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

    # Comment out since already done
    # process_image_folder(RAW_DATA_FOLDER, PROCESSED_DATA_FOLDER)

    # somehow, the script missed these two folders
    # 34 Impatiens
    IMPATIENS_RAW_FOLDER = RAW_DATA_FOLDER / "34Impatiens balsamina(IB)"
    IMPATIENS_PROCESSED_FOLDER = PROCESSED_DATA_FOLDER / "34Impatiens balsamina(IB)"
    # 35 Arachis
    ARACHIS_RAW_FOLDER = RAW_DATA_FOLDER / "35Arachis hypogaea(AH)"
    ARACHIS_PROCESSED_FOLDER = PROCESSED_DATA_FOLDER / "35Arachis hypogaea(AH)"

    print(IMPATIENS_RAW_FOLDER)

    # # Process Impatiens folder
    # process_image_folder(ARACHIS_RAW_FOLDER, ARACHIS_PROCESSED_FOLDER)

    # # Print image counts
    # raw_count = (
    #     len(list(ARACHIS_RAW_FOLDER.glob("*.[jJ][pP][gG]")))
    #     + len(list(ARACHIS_RAW_FOLDER.glob("*.[jJ][pP][eE][gG]")))
    #     + len(list(ARACHIS_RAW_FOLDER.glob("*.webp")))
    #     + len(list(ARACHIS_RAW_FOLDER.glob("*.[pP][nN][gG]")))
    # )
    # processed_count = len(list(ARACHIS_PROCESSED_FOLDER.glob("*.webp")))
    # print(f"Arachis raw images: {raw_count}")
    # print(f"Arachis processed images: {processed_count}")
