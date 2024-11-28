from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Union
import pandas as pd
import umap
from sklearn.pipeline import Pipeline


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
        params = {}
        for param_name, suggest_fn in search_space.items():
            params[param_name] = suggest_fn(trial)
        
        if isinstance(model, Pipeline):
            model['classifier'].set_params(**params)
        else:
            model.set_params(**params)

        # Perform cross-validation and return the mean score
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
        return scores.mean()

    return objective