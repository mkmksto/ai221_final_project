"""Classical Machine Learning Utilities"""

import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def feature_reduction(X, y, method, n_components, **kwargs):
    contain_y = False

    # Initialize the reduction technique with its specific parameters
    if method.lower() == 'umap':
        # Initialize UMAP with any passed kwargs
        reduction_init = umap.UMAP(n_neighbors=kwargs.get('n_neighbors', 15), 
                                   n_components=n_components, 
                                   **kwargs)
        contain_y = True
    elif method.lower() == 'pca':
        # Initialize PCA with any passed kwargs
        reduction_init = PCA(n_components=n_components, **kwargs)
        contain_y = False  # PCA does not require y
    elif method.lower() == 'lda':
        # Initialize LDA with any passed kwargs
        reduction_init = LDA(n_components=n_components, **kwargs)
        contain_y = True  # LDA requires y
    elif method.lower() == 'tsne':
        # Initialize t-SNE with any passed kwargs
        reduction_init = TSNE(n_components=n_components, **kwargs)
        contain_y = False  # t-SNE does not require y

    # Fit and transform the data
    if contain_y and y is not None:
        x_transformed = reduction_init.fit_transform(X, y)
    else:
        x_transformed = reduction_init.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test