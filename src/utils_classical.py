from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from typing import List, Tuple, Union
import pandas as pd
import umap


def feature_reduction(
    X: pd.DataFrame,
    y: Union[List, pd.Series],
    method: str,
    n_components: int,
    n_neighbors: int = 5,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, List, List]:
    """
    Reduces the dimensionality of the dataset using the specified reduction method.

    Parameters:
        X (pd.DataFrame): Features to be reduced.
        y (list): Target labels. Only required for LDA.
        method (str): Dimensionality reduction method ('umap', 'pca', 'lda', or 'tsne').
        n_components (int): Number of components to reduce to.
        n_neighbors (int, optional): Number of neighbors for UMAP (default is 5).
        **kwargs: Additional parameters for the reduction technique.

    Returns:
        tuple: Split datasets - (X_train, X_test, y_train, y_test).
    """
    # Initialize variables
    reduction_init = None
    contain_y = False

    # Choose the reduction method
    method = method.lower()
    if method == 'umap':
        reduction_init = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components)
    elif method == 'pca':
        reduction_init = PCA(n_components=n_components)
    elif method == 'lda':
        reduction_init = LDA(n_components=n_components)
        contain_y = True  # LDA requires y
    elif method == 'tsne':
        reduction_init = TSNE(n_components=n_components)
    else:
        raise ValueError(f"Unsupported method: {method}. Choose from 'umap', 'pca', 'lda', or 'tsne'.")

    # Fit and transform the data
    if contain_y:
        if y is None:
            raise ValueError("Target labels (y) must be provided for LDA.")
        x_transformed = reduction_init.fit_transform(X, y)
    else:
        x_transformed = reduction_init.fit_transform(X)

    if kwargs.get('for_plotting', False):
        return x_transformed, pd.DataFrame, y, pd.DataFrame

    # Split the transformed data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        x_transformed, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test


def split_data(X, y, test_size=0.3):
    return train_test_split(X, y, test_size=test_size, random_state=42)


def generate_objective(X_train, y_train, model, search_space, scoring="accuracy"):
    """
    Generates an Optuna objective function dynamically.
    
    Args:
    - X_train: Training features.
    - y_train: Training labels.
    - model: The input model or pipeline.
    - search_space: Dictionary defining the hyperparameter search space.
    - scoring: Scoring metric for evaluation.

    Returns:
    - A callable Optuna objective function.
    """
    def objective(trial):
        # Dynamically suggest parameters based on the search space
        params = {key: trial.suggest_categorical(key, values) if isinstance(values, list) else trial.suggest_float(key, *values) for key, values in search_space.items()}
        
        if isinstance(model, Pipeline):
            model['classifier'].set_params(**params)
        else:
            model.set_params(**params)

        # Perform cross-validation and return the mean score
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
        return scores.mean()

    return objective

def compare_classification_models(X, y, models, cv=5):
    """
    Compares different classification models on the given dataset.
    
    Parameters:
    - X: Features (e.g., reduced-dimension data from an autoencoder).
    - y: Target labels.
    - models: A dictionary of model names as keys and instantiated classifiers as values.
    - cv: Number of cross-validation folds (default=5).
    
    Returns:
    - A DataFrame with model names and their mean cross-validated accuracy.
    """
    results = []
    
    for model_name, model in models.items():
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        mean_accuracy = scores.mean()
        results.append({'Model': model_name, 'Accuracy': mean_accuracy})
        print(f"Completed {model_name}: Mean Accuracy = {mean_accuracy:.4f}")
    
    # Create a DataFrame to display the results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    
    return results_df
