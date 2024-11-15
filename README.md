# Hello World 🙀🙀

## Dataset

This project uses a subset of the dataset provided by the [Philippine Medicinal Plant Leaf Dataset](https://data.mendeley.com/datasets/tsvdyhbphs/1).

In order to reduce the size of the dataset, the images were resized to 500x500 and converted to the webp format, see `src/utils_image_conversion.py` for the conversion script. The converted dataset is stored in the `data/ph_med_plants_reduced_sizes` folder.

This reduces the size of the dataset to around 30MB from 8GB.

## Introduction

We train various Machine Learning models (mostly classical ML models, and simple MLPs) to classify the images into their corresponding classes.  
The dataset is composed of images of 40 types Philippine medicinal plants (i.e. 40 classes) in various orientations, including both the front and the back part of the leaves.

## Suggested Steps/Pipeline

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
