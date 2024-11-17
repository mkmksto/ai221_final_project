"""Utilities for data handling and EDA."""

import random

import matplotlib.pyplot as plt
import numpy as np

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
