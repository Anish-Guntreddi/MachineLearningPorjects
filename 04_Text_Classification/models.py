"""
Model architectures for text classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    DistilBertForSequenceClassification,
    AlbertForSequenceClassification,
    XLNetForSequenceClassification
)
from typing import Optional, Dict
import numpy as np


class TransformerClassifier(nn.Module):
    """Custom transformer-based classifier with additional layers"""
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_classes: int = 2,
        dropout: float = 0.3,
        hidden_dim: int = 768,
        freeze_backbone: bool = False
    ):
        super(TransformerClassifier, self).__init__()
        
        # Load pretrained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Get hidden size from config
        self.hidden_size = self.transformer.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


class CNNTextClassifier(nn.Module):
    """CNN-based text classifier"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_classes: int = 2,
        num_filters: int = 100,
        filter_sizes: list = [3, 4, 5],
        dropout: float = 0.5,
        pretrained_embeddings: Optional[np.ndarray] = None
    ):
        super(CNNTextClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = False
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim))
            for fs in filter_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, embedding_dim)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch_size, num_filters, conv_seq_len, 1)
            pooled = F.max_pool1d(conv_out.squeeze(3), conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))
        
        # Concatenate all conv outputs
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        
        # Dropout and classification
        x = self.dropout(x)
        logits = self.fc(x)
        
        return logits


class LSTMClassifier(nn.Module):
    """LSTM-based text classifier"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[np.ndarray] = None
    ):
        super(LSTMClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = False
        
        # LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)
    
    def forward(self, x, lengths=None):
        # x shape: (batch_size, sequence_length)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Pack sequence if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Unpack if packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        
        # Use last hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Dropout and classification
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        
        return logits


class AttentionClassifier(nn.Module):
    """Text classifier with self-attention mechanism"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: int = 2,
        max_length: int = 512,
        dropout: float = 0.1
    ):
        super(AttentionClassifier, self).__init__()
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Embedding(max_length, embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings with positional encoding
        x = self.embedding(x) + self.positional_encoding(positions)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Global average pooling
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            x = (x * (1 - mask)).sum(dim=1) / (1 - mask).sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class EnsembleClassifier(nn.Module):
    """Ensemble of multiple text classifiers"""
    
    def __init__(self, models: list, weights: Optional[list] = None):
        super(EnsembleClassifier, self).__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
    
    def forward(self, *args, **kwargs):
        outputs = []
        for model in self.models:
            output = model(*args, **kwargs)
            outputs.append(output)
        
        # Weighted average of logits
        weighted_output = torch.zeros_like(outputs[0])
        for output, weight in zip(outputs, self.weights):
            weighted_output += weight * output
        
        return weighted_output


def get_model(
    model_name: str,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to get text classification model
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments
    
    Returns:
        Text classification model
    """
    model_name = model_name.lower()
    
    if model_name == 'bert':
        return BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_classes
        )
    elif model_name == 'roberta':
        return RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=num_classes
        )
    elif model_name == 'distilbert':
        return DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=num_classes
        )
    elif model_name == 'albert':
        return AlbertForSequenceClassification.from_pretrained(
            'albert-base-v2',
            num_labels=num_classes
        )
    elif model_name == 'xlnet':
        return XLNetForSequenceClassification.from_pretrained(
            'xlnet-base-cased',
            num_labels=num_classes
        )
    elif model_name == 'custom_transformer':
        return TransformerClassifier(num_classes=num_classes, **kwargs)
    elif model_name == 'cnn':
        return CNNTextClassifier(num_classes=num_classes, **kwargs)
    elif model_name == 'lstm':
        return LSTMClassifier(num_classes=num_classes, **kwargs)
    elif model_name == 'attention':
        return AttentionClassifier(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test TransformerClassifier
    model = TransformerClassifier(num_classes=2).to(device)
    input_ids = torch.randint(0, 1000, (2, 128)).to(device)
    attention_mask = torch.ones(2, 128).to(device)
    
    output = model(input_ids, attention_mask)
    print(f"TransformerClassifier output shape: {output.shape}")
    
    # Test CNN
    model = CNNTextClassifier(vocab_size=10000, num_classes=4).to(device)
    x = torch.randint(0, 10000, (2, 100)).to(device)
    output = model(x)
    print(f"CNN output shape: {output.shape}")
    
    # Test LSTM
    model = LSTMClassifier(vocab_size=10000, num_classes=4).to(device)
    output = model(x)
    print(f"LSTM output shape: {output.shape}")