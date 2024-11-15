"""Hello World"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


class SampleNN(nn.Module):
    """Sample NN architecture"""

    def __init__(
        self,
        input_size: int = 561,
        hidden_sizes: list[int] = [256, 128],
        output_size: int = 6,
    ) -> None:
        super(SampleNN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers = []
        prev_size = input_size

        # create hidden layers
        for size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, size),
                    nn.ReLU(),
                    nn.BatchNorm1d(size),
                    # try changing this as well
                    nn.Dropout(0.2),
                ]
            )
            prev_size = size

        # create output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)

        # force move model to CPU
        self.model = self.model.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network."""
        return self.model(x)


def test_function():
    """Test function"""
    print("Hello World from `src/utils_har.py`")
