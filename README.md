# Hello World 🙀🙀

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
├── data
│ └── ph_med_plants_reduced_sizes
│ ├── 10Coleus scutellarioides(CS)
│ ├── ...
│ └── 9Centella asiatica(CA)
├── models
│ └── test_model
├── notebooks
│ ├── anila_test_1.ipynb
│ ├── cantor_test_1.ipynb
│ └── quinto_test_1.ipynb
├── README.md
├── requirements.txt
├── src
│ ├── utils_classical.py
│ ├── utils_image_conversion.py
│ ├── utils_nn.py
│ └── utils_preprocessing.py
└── TODOs.md
```

- where `create_submission.sh` is a script to create a compressed submission file for the project.
- `data` contains the original and processed dataset.
- `models` contains the trained models.
- `notebooks` contains the notebooks used for testing and development.
- `src` contains the utility functions for the project.

## General Suggested Steps/Pipeline

1. Data Loading and Initial EDA

   - Load and organize the dataset into train/validation/test splits
   - Analyze class distribution to check for imbalances
   - Visualize sample images from each class
   - Calculate basic statistics (image sizes, color distributions, etc.)

2. Feature Extraction

   - Extract relevant features from images using techniques like:
     - HOG (Histogram of Oriented Gradients)
     - SIFT (Scale-Invariant Feature Transform)
     - Color histograms
     - Texture features (GLCM)
   - Dimensionality reduction if needed (PCA, t-SNE, UMAP, etc.)
   - try din natin i-implement ang autoencoders?

3. Data Augmentation Pipeline

   - Implement on-the-fly augmentations:
     - Random rotations (0-360 degrees)
     - Horizontal and vertical flips
     - Random crops (with padding)
     - Brightness/contrast adjustments
     - Random noise addition
     - Color jittering

4. Classical ML Models Implementation
   Suggested models:

   - Support Vector Machine (SVM) with RBF kernel
   - Random Forest Classifier
   - XGBoost
   - K-Nearest Neighbors (KNN)
     These models work well with image features and can handle high-dimensional data

5. Neural Network Implementation

   - Design a simple MLP using PyTorch:
     - Input layer based on extracted features
     - 2-3 hidden layers with ReLU activation
     - Dropout for regularization
     - Softmax output layer for 40 classes
   - Implement training loop with early stopping

6. Model Training and Validation

   - Train all models using augmented training data
   - Use cross-validation for classical ML models
   - Monitor validation metrics during training
   - Save model checkpoints for best performers

7. Model Evaluation and Comparison

   - Evaluate models using:
     - Accuracy
     - Precision, Recall, F1-score
     - Confusion matrices
     - ROC curves for multi-class scenario
   - Analyze per-class performance
   - Compare computational requirements

8. Error Analysis

   - Identify common misclassifications
   - Analyze challenging cases
   - Visualize activation/feature maps for neural network
   - Generate insights for potential improvements

9. Results Visualization and Reporting
   - Create comparative plots of model performances
   - Generate detailed classification reports
   - Visualize decision boundaries where applicable
   - Document findings and insights
   - Provide recommendations for model selection

General Todos

- [ ] Classical ML models
- [ ] Neural Networks (MLP)
- [ ] Paper in LaTeX?
- [ ] Presentation Board (includes the full pipeline, and other relevant diagrams)

# Detailed Image Processing and Augmentation Pipeline

## 1. Preprocessing Pipeline

### Basic Preprocessing

- Resize images to consistent size (already done at 500x500)
- Convert to grayscale for certain feature extractors
- Normalize pixel values (0-1 or -1 to 1)
- Remove background noise using thresholding techniques

### Feature Extraction

Especially important for classical ML:

- Edge detection (already implemented using Canny)
- HOG (Histogram of Oriented Gradients) - great for leaf shape analysis
- Color histograms (RGB/HSV) - useful for leaf color variations
- GLCM (Gray Level Co-occurrence Matrix) - for texture analysis
- SIFT/SURF features - for distinctive leaf patterns

### Advanced Preprocessing

- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Bilateral filtering to preserve edges while reducing noise
- Segmentation to isolate leaf from background

## 2. Augmentation Pipeline

Using Albumentations would be ideal as it's faster than PIL and provides more options:

### Geometric Transformations

- Random rotations (0-360° since leaves can be in any orientation)
- Horizontal/Vertical flips
- Random scaling (±20%)
- Random cropping with padding

### Color/Intensity Transformations

- Brightness/Contrast adjustments
- Color jittering
- Random gamma
- Channel shuffling
- Random shadow/highlights

### Noise/Blur

- Gaussian noise
- Motion blur
- Gaussian blur
- ISO noise

## 3. Using PyTorch DataLoader with Classical ML

### Custom Dataset Implementation

1. Create a custom Dataset class that:
   - Loads images
   - Applies preprocessing
   - Returns both the processed image features and labels
   - Includes an option to return either numpy arrays (for sklearn) or tensors (for PyTorch)

### DataLoader Configuration

2. Use DataLoader with:
   - `num_workers` for parallel processing
   - `batch_size=1` for classical ML during training
   - Larger batch sizes for feature extraction phase

### Key Benefits

The key is to convert the batched tensor output from DataLoader back to numpy arrays when using with sklearn models.

This approach gives you:

- Consistent data loading pipeline for both deep learning and classical ML
- Parallel data loading and preprocessing
- Memory efficiency through batching
- Easy switching between models

# Important Notes for the Preprocessing and Augmentation Pipeline

- the pipelines are a bit different for classical ML and neural networks.
- For Classical ML, we train on the enhanced images with feature extraction, this is because classical ML models aren't as good as neural networks at extracting features.
- Also for Classical ML, we first start without data augmentation and only add it later if we see that the model is over or underfitting.
- suggested preprocessing for classical ML:
  - edge detection (already implemented using Canny)
  - HOG (Histogram of Oriented Gradients) - great for leaf shape analysis
  - Color histograms (RGB/HSV) - useful for leaf color variations
  - GLCM (Gray Level Co-occurrence Matrix) - for texture analysis
  - basic denoising
  - background removal

## Classical ML Pipeline

- Original Image → Basic Preprocessing → Augmentation → Feature Extraction (Resize, (Rotations, (HOG, SIFT, Basic Denoising) Flips, etc.) Color Histograms)

## Neural Network Pipeline

- Original Image → Basic Preprocessing → Augmentation → Normalization (Resize, (Rotations, (Pixel scaling, Basic Denoising) Flips, etc.) Channel means)

## Things common to both pipelines

- Best practice: Augmentations should typically be performed on the original (or more likely: the minimally preprocessed) images, not on the heavily preprocessed images.
