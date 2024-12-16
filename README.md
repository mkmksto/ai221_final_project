# AI 221 Final Project

## Dataset

This project uses a subset of the dataset provided by the [Philippine Medicinal Plant Leaf Dataset](https://data.mendeley.com/datasets/tsvdyhbphs/1).

In order to reduce the size of the dataset, the images were resized to 500x500 and converted to the webp format, see `src/utils_image_conversion.py` for the conversion script. The converted dataset is stored in the `data/ph_med_plants_reduced_sizes` folder.

This reduces the size of the dataset to around 30MB from 8GB.

## Introduction

We train various Machine Learning models (mostly classical ML models, and simple MLPs) to classify the images into their corresponding classes.  
The dataset is composed of images of 40 types Philippine medicinal plants (i.e. 40 classes) in various orientations, including both the front and the back part of the leaves.

## Project Structure

The project structure is as follows:

```
.
├── create_submission.sh
├── models
├── notebooks
│   ├── bg_removal.ipynb
│   ├── cantor_official_classification.ipynb
│   ├── cantor_test_1.ipynb
│   ├── cantor_training_autoencoder.ipynb
│   ├── cantor_training.ipynb
│   ├── quinto_automl.ipynb
│   ├── quinto_eda.ipynb
│   ├── quinto_etc.ipynb
│   └── quinto_training.ipynb
├── README.md
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── utils_autoencoder.py
│   ├── utils_classical.py
│   ├── utils_data.py
│   ├── utils_image_conversion.py
│   ├── utils_nn.py
│   ├── utils_plotting.py
│   └── utils_preprocessing.py
└── TODOs.md
```

- where `create_submission.sh` is a script to create a compressed submission file for the project.
- `data` contains the original and processed dataset.
- `models` contains the trained models.
- `notebooks` contains the notebooks used for testing and development.
- `src` contains the utility functions for the project.
