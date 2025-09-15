"""
Model architectures for Anomaly Detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np


class AutoEncoder(nn.Module):
    """AutoEncoder for anomaly detection"""
    
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 32,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu',
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # Choose activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.LeakyReLU(0.2)
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = encoding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


class VariationalAutoEncoder(nn.Module):
    """Variational AutoEncoder for anomaly detection"""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 20,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_var = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var


class LSTMAutoEncoder(nn.Module):
    """LSTM AutoEncoder for time series anomaly detection"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Latent representation
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder LSTM
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.output_fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Encode
        _, (hidden, cell) = self.encoder_lstm(x)
        
        # Get latent representation from last hidden state
        latent = self.encoder_fc(hidden[-1])
        
        # Decode
        hidden_decoded = self.decoder_fc(latent)
        hidden_decoded = hidden_decoded.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Initialize decoder hidden state
        h_0 = hidden
        c_0 = cell
        
        output, _ = self.decoder_lstm(hidden_decoded, (h_0, c_0))
        output = self.output_fc(output)
        
        return output


class IsolationForestNN(nn.Module):
    """Neural network-based Isolation Forest"""
    
    def __init__(
        self,
        input_dim: int,
        n_estimators: int = 100,
        max_samples: int = 256
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        
        # Create ensemble of isolation trees
        self.trees = nn.ModuleList([
            IsolationTree(input_dim) for _ in range(n_estimators)
        ])
    
    def forward(self, x):
        # Get anomaly scores from all trees
        scores = []
        for tree in self.trees:
            score = tree(x)
            scores.append(score)
        
        # Average scores
        scores = torch.stack(scores, dim=1)
        anomaly_score = torch.mean(scores, dim=1)
        
        return anomaly_score


class IsolationTree(nn.Module):
    """Single isolation tree"""
    
    def __init__(self, input_dim: int, max_depth: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.max_depth = max_depth
        
        # Neural network to learn splitting
        self.split_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Compute isolation score
        path_length = self.split_network(x).squeeze()
        
        # Normalize to anomaly score
        n = x.shape[0]
        c_n = 2 * (np.log(n - 1) + 0.5772) - 2 * (n - 1) / n if n > 1 else 1
        anomaly_score = 2 ** (-path_length / c_n)
        
        return anomaly_score


class OneClassSVM(nn.Module):
    """Neural network-based One-Class SVM"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        kernel: str = 'rbf',
        nu: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel = kernel
        self.nu = nu
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decision boundary
        self.decision_boundary = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Apply kernel if specified
        if self.kernel == 'rbf':
            features = torch.exp(-torch.norm(features, dim=1, keepdim=True) ** 2)
        
        # Decision function
        decision = self.decision_boundary(features)
        
        return decision.squeeze()


class DeepSVDD(nn.Module):
    """Deep Support Vector Data Description"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        latent_dim: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.center = None
        
        # Build encoder
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        z = self.encoder(x)
        
        if self.center is None:
            # Initialize center with mean of first batch
            self.center = torch.mean(z, dim=0).detach()
        
        # Calculate distance to center
        dist = torch.sum((z - self.center) ** 2, dim=1)
        
        return z, dist
    
    def get_anomaly_score(self, x):
        _, dist = self.forward(x)
        return dist


class GANomaly(nn.Module):
    """GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training"""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 100,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Generator (Encoder-Decoder-Encoder)
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
        self.encoder2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encode
        z = self.encoder1(x)
        
        # Decode
        x_hat = self.decoder(z)
        
        # Re-encode
        z_hat = self.encoder2(x_hat)
        
        # Discriminate
        real_score = self.discriminator(x)
        fake_score = self.discriminator(x_hat)
        
        return x_hat, z, z_hat, real_score, fake_score
    
    def get_anomaly_score(self, x):
        z = self.encoder1(x)
        x_hat = self.decoder(z)
        z_hat = self.encoder2(x_hat)
        
        # Anomaly score based on reconstruction and encoding errors
        reconstruction_error = torch.mean((x - x_hat) ** 2, dim=1)
        encoding_error = torch.mean((z - z_hat) ** 2, dim=1)
        
        anomaly_score = reconstruction_error + encoding_error
        
        return anomaly_score


def get_model(
    model_name: str,
    input_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to get anomaly detection model
    
    Args:
        model_name: Name of the model
        input_dim: Input dimension
        **kwargs: Additional model-specific arguments
    
    Returns:
        Anomaly detection model
    """
    model_name = model_name.lower()
    
    if model_name == 'autoencoder' or model_name == 'ae':
        return AutoEncoder(input_dim=input_dim, **kwargs)
    elif model_name == 'vae':
        return VariationalAutoEncoder(input_dim=input_dim, **kwargs)
    elif model_name == 'lstm_ae':
        return LSTMAutoEncoder(input_dim=input_dim, **kwargs)
    elif model_name == 'isolation_forest':
        return IsolationForestNN(input_dim=input_dim, **kwargs)
    elif model_name == 'ocsvm':
        return OneClassSVM(input_dim=input_dim, **kwargs)
    elif model_name == 'deep_svdd':
        return DeepSVDD(input_dim=input_dim, **kwargs)
    elif model_name == 'ganomaly':
        return GANomaly(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 32
    input_dim = 10
    seq_len = 50
    
    # Test AutoEncoder
    model = AutoEncoder(input_dim=input_dim).to(device)
    x = torch.randn(batch_size, input_dim).to(device)
    output = model(x)
    print(f"AutoEncoder output shape: {output.shape}")
    
    # Test VAE
    model = VariationalAutoEncoder(input_dim=input_dim).to(device)
    output, mu, log_var = model(x)
    print(f"VAE output shape: {output.shape}")
    
    # Test LSTM AutoEncoder
    model = LSTMAutoEncoder(input_dim=input_dim).to(device)
    x_seq = torch.randn(batch_size, seq_len, input_dim).to(device)
    output = model(x_seq)
    print(f"LSTM AutoEncoder output shape: {output.shape}")
    
    # Test Deep SVDD
    model = DeepSVDD(input_dim=input_dim).to(device)
    z, dist = model(x)
    print(f"Deep SVDD latent shape: {z.shape}, distance shape: {dist.shape}")
    
    # Test GANomaly
    model = GANomaly(input_dim=input_dim).to(device)
    x_hat, z, z_hat, real_score, fake_score = model(x)
    print(f"GANomaly reconstruction shape: {x_hat.shape}")
    anomaly_scores = model.get_anomaly_score(x)
    print(f"Anomaly scores shape: {anomaly_scores.shape}")