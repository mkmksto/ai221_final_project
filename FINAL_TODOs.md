# Final Todos for submission

## Code

- [x] VGG16 model as baseline

### ML models (raw, without hyperparam optimization)

- [x] autosklearn (best performing model: SVM/SVC, accuracy: 0.98)
- [x] SVM (https://github.com/mkmksto/ai221_final_project/blob/main/notebooks/cantor_training.ipynb)
- [x] XGBoost
- [x] Random Forest (0.97)
- [x] K-Nearest Neighbors (0.969)
- [x] MLP (0.91)

### ML Models with dim reduction (e.g. Autoencoder)

Only perform on maybe the top 3 best models from the previous section

- [x] SVM + AE (https://github.com/mkmksto/ai221_final_project/blob/main/notebooks/cantor_official_classification.ipynb)
- [x] KNN + AE
- [x] MLP + AE
- [x] Logistic Regression + AE
- [x] Random Forest + AE (https://github.com/mkmksto/ai221_final_project/blob/main/notebooks/cantor_training_autoencoder.ipynb)
- [x] Decision Trees + AE
- [x] XGBoost + LDA (0.949)
- [x] XGBoost + AE (0.916)

### Hyperparam optimized models

- [x] SVM + AE + Optuna
- [x] KNN + AE + Optuna
- [x] MLP + AE + Optuna
- [x] XGBoost + LDA + Optuna (0.959) (and 0.977 if no dim reduction)

## Tables

Metrics: Classification Report (Accuracy, Precision, Recall, F1 Score)
Note: only use accuracy for most models, only add the classification report for the top 3 best performing models with hyperparam tuning

- Table Comparing all the models (raw, with and without dim reduction)
- Table Comparing all the models (hyperparam optimized)
- Table Comparing all the models (dim reduction only)

## Figures (for the top 3 best performing models)

- [ ] Confusion Matrix
- [x] t-SNE Plot
- [x] UMAP Plot

## Flowcharts

- [x] Data Preparation Pipeline
  - [ ] Resizing and Conversion to webp
  - [ ] Background Removal
  - [ ] Feature Extraction
- [x] Training Pipeline
  - [ ] Model Selection
  - [ ] With or without dim reduction
  - [ ] Hyperparam optimization
- [x] Inference Pipeline
  - [ ] BG removal and resizing
  - [ ] Other preprocessing steps
  - [ ] Feature Extraction
  - [ ] Model Inference

## Slides

- [ ] Match the structure of the paper

## Paper

- 2 parts: Executive Summary (2 pages) and Supporting Details (any number of pages)

## Graphical Abstract

- [ ] To be done in figma or photoshop
- [x] Data Preparation Pipeline
- [x] Training Pipeline
- [x] Inference Pipeline
- [ ] Performance Metrics
- [ ] Additional: PFI
