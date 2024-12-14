# Final Todos for submission

## Code

- [x] VGG16 model as baseline

### ML models (raw, without hyperparam optimization)

- [ ] autosklearn
- [x] SVM (https://github.com/mkmksto/ai221_final_project/blob/main/notebooks/cantor_training.ipynb)
- [x] XGBoost
- [ ] Random Forest
- [ ] K-Nearest Neighbors
- [ ] MLP

### ML Models with dim reduction (e.g. Autoencoder)

Only perform on maybe the top 3 best models from the previous section

- [x] SVM + AE (https://github.com/mkmksto/ai221_final_project/blob/main/notebooks/cantor_official_classification.ipynb)
- [x] KNN + AE
- [x] MLP + AE
- [x] Logistic Regression + AE
- [x] Random Forest + AE (https://github.com/mkmksto/ai221_final_project/blob/main/notebooks/cantor_training_autoencoder.ipynb)
- [x] Decision Trees + AE
- [ ] XGBoost + PCA/LDA/AE

### Hyperparam optimized models

- [x] SVM + AE + Optuna
- [x] KNN + AE + Optuna
- [x] MLP + AE + Optuna
- [ ] XGBoost + AE + Optuna

## Tables

Metrics: Classification Report (Accuracy, Precision, Recall, F1 Score)

- Table Comparing all the models (raw, with and without dim reduction)
- Table Comparing all the models (hyperparam optimized)
- Table Comparing all the models (dim reduction only)

## Figures

- [ ] Confusion Matrix
- [ ] t-SNE Plot
- [ ] UMAP Plot

## Flowcharts

- [ ] Data Preparation Pipeline
  - [ ] Resizing and Conversion to webp
  - [ ] Background Removal
  - [ ] Feature Extraction
- [ ] Training Pipeline
  - [ ] Model Selection
  - [ ] With or without dim reduction
  - [ ] Hyperparam optimization
- [ ] Inference Pipeline
  - [ ] BG removal and resizing
  - [ ] Other preprocessing steps
  - [ ] Feature Extraction
  - [ ] Model Inference
