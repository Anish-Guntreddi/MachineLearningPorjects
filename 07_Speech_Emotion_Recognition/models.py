"""
Model architectures for Speech Emotion Recognition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class CNN1D(nn.Module):
    """1D CNN for raw audio or feature sequences"""
    
    def __init__(
        self,
        input_dim: int = 1,
        num_classes: int = 8,
        hidden_dims: list = [64, 128, 256, 512],
        kernel_sizes: list = [7, 5, 3, 3],
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.conv_blocks = nn.ModuleList()
        in_channels = input_dim
        
        for hidden_dim, kernel_size in zip(hidden_dims, kernel_sizes):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout)
                )
            )
            in_channels = hidden_dim
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, channels, time)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class CNN2D(nn.Module):
    """2D CNN for spectrograms"""
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 8,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network"""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        
        # RNN
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # CNN features
        cnn_out = self.cnn(x)
        
        # Transpose for RNN (batch, time, features)
        cnn_out = cnn_out.transpose(1, 2)
        
        # RNN
        rnn_out, _ = self.rnn(cnn_out)
        
        # Attention
        attention_weights = self.attention(rnn_out)
        attention_weights = F.softmax(attention_weights.squeeze(-1), dim=1)
        attended_features = torch.sum(
            rnn_out * attention_weights.unsqueeze(-1),
            dim=1
        )
        
        # Classification
        output = self.classifier(attended_features)
        
        return output


class TransformerModel(nn.Module):
    """Transformer-based emotion recognition model"""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 8,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.5,
        max_seq_length: int = 1000
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, x):
        # x shape: (batch, features, time)
        x = x.transpose(1, 2)  # (batch, time, features)
        
        # Project input
        x = self.input_projection(x)
        
        # Add CLS token
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use CLS token for classification
        cls_output = x[:, 0]
        
        # Classification
        output = self.classifier(cls_output)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    """Attention-based pooling layer"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, time, features)
        attention_weights = self.attention(x)
        attention_weights = F.softmax(attention_weights.squeeze(-1), dim=1)
        weighted_sum = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        return weighted_sum


class HybridModel(nn.Module):
    """Hybrid model combining CNN and Transformer"""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 8,
        cnn_channels: list = [64, 128, 256],
        d_model: int = 256,
        nhead: int = 8,
        num_transformer_layers: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # CNN feature extractor
        cnn_layers = []
        in_channels = input_dim
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Project CNN output to transformer dimension
        self.projection = nn.Linear(cnn_channels[-1], d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers)
        
        # Attention pooling
        self.attention_pool = AttentionPooling(d_model)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # CNN features
        cnn_features = self.cnn(x)
        
        # Transpose for transformer (batch, time, features)
        cnn_features = cnn_features.transpose(1, 2)
        
        # Project to transformer dimension
        projected = self.projection(cnn_features)
        
        # Transformer encoding
        transformer_out = self.transformer(projected)
        
        # Attention pooling
        pooled = self.attention_pool(transformer_out)
        
        # Classification
        output = self.classifier(pooled)
        
        return output


def get_model(
    model_name: str,
    input_dim: int,
    num_classes: int = 8,
    **kwargs
) -> nn.Module:
    """
    Factory function to get emotion recognition model
    
    Args:
        model_name: Name of the model
        input_dim: Input dimension
        num_classes: Number of emotion classes
        **kwargs: Additional model-specific arguments
    
    Returns:
        Emotion recognition model
    """
    model_name = model_name.lower()
    
    if model_name == 'cnn1d':
        return CNN1D(input_dim=input_dim, num_classes=num_classes, **kwargs)
    elif model_name == 'cnn2d':
        return CNN2D(input_channels=1 if input_dim == 1 else input_dim, 
                     num_classes=num_classes, **kwargs)
    elif model_name == 'crnn':
        return CRNN(input_dim=input_dim, num_classes=num_classes, **kwargs)
    elif model_name == 'transformer':
        return TransformerModel(input_dim=input_dim, num_classes=num_classes, **kwargs)
    elif model_name == 'hybrid':
        return HybridModel(input_dim=input_dim, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test CNN1D
    model = CNN1D(input_dim=40, num_classes=8).to(device)
    x = torch.randn(2, 40, 100).to(device)  # (batch, features, time)
    output = model(x)
    print(f"CNN1D output shape: {output.shape}")
    
    # Test CNN2D
    model = CNN2D(input_channels=1, num_classes=8).to(device)
    x = torch.randn(2, 1, 128, 128).to(device)  # (batch, channels, height, width)
    output = model(x)
    print(f"CNN2D output shape: {output.shape}")
    
    # Test CRNN
    model = CRNN(input_dim=40, num_classes=8).to(device)
    x = torch.randn(2, 40, 200).to(device)
    output = model(x)
    print(f"CRNN output shape: {output.shape}")
    
    # Test Transformer
    model = TransformerModel(input_dim=40, num_classes=8).to(device)
    x = torch.randn(2, 40, 100).to(device)
    output = model(x)
    print(f"Transformer output shape: {output.shape}")
    
    # Test Hybrid
    model = HybridModel(input_dim=40, num_classes=8).to(device)
    x = torch.randn(2, 40, 200).to(device)
    output = model(x)
    print(f"Hybrid output shape: {output.shape}")