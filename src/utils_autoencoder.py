import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SemiSupervisedAutoencoder:
    def __init__(self, input_dim, encoding_dim, num_classes, dropout=0.4, learning_rate=0.001):
        # Initialize the Autoencoder model
        self.autoencoder = self._build_autoencoder(input_dim, encoding_dim, num_classes, dropout)
        self.criterion_reconstruction = nn.MSELoss()
        self.criterion_classification = nn.CrossEntropyLoss()
        self.optimizer_autoencoder = optim.Adam(self.autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.encoding_dim = encoding_dim

    def _build_autoencoder(self, input_dim, encoding_dim, num_classes, dropout):
        class Autoencoder(nn.Module):
            def __init__(self):
                super(Autoencoder, self).__init__()
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, encoding_dim)
                )
                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, input_dim),
                    nn.Sigmoid()
                )
                # Classifier
                self.classifier = nn.Sequential(
                    nn.Linear(encoding_dim, num_classes)
                )

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                classification = self.classifier(encoded)
                return encoded, decoded, classification
        return Autoencoder()

    def extract_features(self, dataloader):
        """Extract features from the encoder (latent space)."""
        self.autoencoder.eval()
        features = []
        with torch.no_grad():
            for batch in dataloader:
                data = batch[0]
                encoded = self.autoencoder.encoder(data)
                features.append(encoded)
        # Concatenate all extracted features into a single tensor
        features = torch.cat(features, dim=0)
        print(f"Extracted features shape: {features.shape}")
        return features

    def train_autoencoder(self, dataloader, epochs=50):
        """Train the autoencoder on unlabeled data."""
        self.autoencoder.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in dataloader:
                data = batch[0]
                self.optimizer_autoencoder.zero_grad()
                _, reconstructed, _ = self.autoencoder(data)
                loss = self.criterion_reconstruction(reconstructed, data)
                loss.backward()
                self.optimizer_autoencoder.step()
                epoch_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Reconstruction Loss: {epoch_loss / len(dataloader):.4f}")

    def train_classifier(self, dataloader, labeled_optimizer, epochs=30):
        """Train the classifier on labeled data (freeze the encoder)."""
        # Freeze the encoder
        for param in self.autoencoder.encoder.parameters():
            param.requires_grad = False

        self.autoencoder.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in dataloader:
                data, labels = batch
                labeled_optimizer.zero_grad()
                encoded = self.autoencoder.encoder(data)
                classification = self.autoencoder.classifier(encoded)
                loss = self.criterion_classification(classification, labels)
                loss.backward()
                labeled_optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Classification Loss: {epoch_loss / len(dataloader):.4f}")

    def fine_tune(self, dataloader, fine_tune_optimizer, epochs=10):
        """Fine-tune the entire model on labeled data."""
        # Unfreeze the encoder
        for param in self.autoencoder.encoder.parameters():
            param.requires_grad = True

        self.autoencoder.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in dataloader:
                data, labels = batch
                fine_tune_optimizer.zero_grad()
                _, _, classification = self.autoencoder(data)
                loss = self.criterion_classification(classification, labels)
                loss.backward()
                fine_tune_optimizer.step()
                epoch_loss += loss.item()
            print(f"Fine-Tune Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

    def evaluate(self, dataloader):
        """Evaluate the classifier."""
        self.autoencoder.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in dataloader:
                _, _, outputs = self.autoencoder(data)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy
    
def create_dataloader(X_labeled, X_unlabeled, y_labeled):
    # Convert to PyTorch tensors
    X_labeled_tensor = torch.tensor(X_labeled, dtype=torch.float32)
    y_labeled_tensor = torch.tensor(y_labeled, dtype=torch.long)
    X_unlabeled_tensor = torch.tensor(X_unlabeled, dtype=torch.float32)

    # Create DataLoaders
    labeled_dataset = TensorDataset(X_labeled_tensor, y_labeled_tensor)
    unlabeled_dataset = TensorDataset(X_unlabeled_tensor)

    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=True)
    labeled_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)

    return unlabeled_loader, labeled_loader

def full_X_dataloader(X):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    return DataLoader(TensorDataset(torch.tensor(X_tensor, dtype=torch.float32)), batch_size=32, shuffle=False)