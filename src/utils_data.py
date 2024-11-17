"""Utilities for data handling and EDA."""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_FOLDER = Path(__file__).parent.parent / "data"
RAW_DATA_FOLDER = DATA_FOLDER / "Philippine Medicinal Plant Leaf Dataset(raw)"
PROCESSED_DATA_FOLDER = DATA_FOLDER / "ph_med_plants_reduced_sizes"

HIBISCUS_RAW_FOLDER = RAW_DATA_FOLDER / "1Hibiscus rosa-sinensis(HRS)"
CARMONA_RETUSA_RAW_FOLDER = RAW_DATA_FOLDER / "16Carmona retusa(CR)"

HIBISCUS_PROCESSED_FOLDER = PROCESSED_DATA_FOLDER / "1Hibiscus rosa-sinensis(HRS)"
CARMONA_RETUSA_PROCESSED_FOLDER = PROCESSED_DATA_FOLDER / "16Carmona retusa(CR)"
AVERRHOEA_BILIMBI_PROCESSED_FOLDER = PROCESSED_DATA_FOLDER / "26Averrhoea bilimbi(AVB)"


LIST_OF_FOLDER_CLASSES = [
    "1Hibiscus rosa-sinensis(HRS)",
    "2Psidium guajava(PG)",
    "3Antidesma bunius(AB)",
    "4Vitex negundo(VN)",
    "5Moringa oleifera(MO)",
    "6Blumea balsamifera(BB)",
    "7Origanum vulgare(OV)",
    "8Pepromia pellucida(PP)",
    "9Centella asiatica(CA)",
    "10Coleus scutellarioides(CS)",
    "11Phyllanthus niruri(PN)",
    "12Corchorus olitorius(CO)",
    "13Momordica charantia (MC)",
    "14Euphorbia hirta(EH)",
    "15Curcuma longa(CL)",
    "16Carmona retusa(CR)",
    "17Senna alata(SA)",
    "18Mentha cordifolia Opiz(MCO)",
    "19Capsicum frutescens(CF)",
    "20Jatropha curcas(JC)",
    "21Ocimum basilicum(OB)",
    "22Nerium oleander(NO)",
    "23Pandanus amaryllifolius(PA)",
    "24Aloe barbadensis Miller(ABM)",
    "25Lagerstroemia speciosa(LS)",
    "26Averrhoea bilimbi(AVB)",
    "27Annona muricata(AM)",
    "28Citrus aurantiifolia(CIA)",
    "29Premna odorata(PO)",
    "30Gliricidia sepium(GS)",
    "31Citrus sinensis(CIS)",
    "32Mangifera indica(MI)",
    "33Citrus microcarpa(CM)",
    "34Impatiens balsamina(IB)",
    "35Arachis hypogaea(AH)",
    "36Tamarindus indica(TI)",
    "37Leucaena leucocephala(LL)",
    "38Ipomoea batatas(IPB)",
    "39Manihot esculenta(ME)",
    "40Citrus maxima(CMA)",
]

# Dictionary mapping class numbers to class names
PLANT_CLASS_DICT: dict[int, str] = {
    1: "Hibiscus rosa-sinensis",
    2: "Psidium guajava",
    3: "Antidesma bunius",
    4: "Vitex negundo",
    5: "Moringa oleifera",
    6: "Blumea balsamifera",
    7: "Origanum vulgare",
    8: "Pepromia pellucida",
    9: "Centella asiatica",
    10: "Coleus scutellarioides",
    11: "Phyllanthus niruri",
    12: "Corchorus olitorius",
    13: "Momordica charantia",
    14: "Euphorbia hirta",
    15: "Curcuma longa",
    16: "Carmona retusa",
    17: "Senna alata",
    18: "Mentha cordifolia Opiz",
    19: "Capsicum frutescens",
    20: "Jatropha curcas",
    21: "Ocimum basilicum",
    22: "Nerium oleander",
    23: "Pandanus amaryllifolius",
    24: "Aloe barbadensis Miller",
    25: "Lagerstroemia speciosa",
    26: "Averrhoea bilimbi",
    27: "Annona muricata",
    28: "Citrus aurantiifolia",
    29: "Premna odorata",
    30: "Gliricidia sepium",
    31: "Citrus sinensis",
    32: "Mangifera indica",
    33: "Citrus microcarpa",
    34: "Impatiens balsamina",
    35: "Arachis hypogaea",
    36: "Tamarindus indica",
    37: "Leucaena leucocephala",
    38: "Ipomoea batatas",
    39: "Manihot esculenta",
    40: "Citrus maxima",
}


class_folders = [
    PROCESSED_DATA_FOLDER / folder_class for folder_class in LIST_OF_FOLDER_CLASSES
]

# per row: class_folder, class_name, class_number
RAW_DATA_DF = pd.DataFrame(
    {
        "class_folder": class_folders,
        "class_name": LIST_OF_FOLDER_CLASSES,
        "class_number": range(1, len(LIST_OF_FOLDER_CLASSES) + 1),
    }
)


def plot_random_images_grid(class_folders: list) -> None:
    """Plot one random image from each class in a grid layout.

    Args:
        class_folders: List of paths to class folders
    """

    # Create a 5x8 grid of subplots showing one random image from each class
    fig, axes = plt.subplots(5, 8, figsize=(15, 8))

    # Plot one random image from each class
    for idx, folder in enumerate(class_folders):
        row = idx // 8
        col = idx % 8

        # Get list of image files and select one randomly
        image_files = list(folder.glob("*.webp"))
        random_image = random.choice(image_files)

        # Read and display image
        img = plt.imread(random_image)
        axes[row, col].imshow(img)
        axes[row, col].axis("off")

        # Add class name as title
        class_name = folder.name.split("_")[0]  # Get first part of folder name
        axes[row, col].set_title(class_name, fontsize=8)

    plt.tight_layout()
    plt.show()
