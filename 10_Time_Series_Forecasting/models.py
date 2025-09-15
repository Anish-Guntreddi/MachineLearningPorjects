"""
Model architectures for Time Series Forecasting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math
import numpy as np


class LSTM(nn.Module):
    """LSTM model for time series forecasting"""
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        prediction_length: int = 12,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layers
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch, sequence_length, input_dim)
        batch_size = x.size(0)
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state for prediction
        if self.bidirectional:
            # Combine forward and backward hidden states
            hidden_forward = hidden[-2]
            hidden_backward = hidden[-1]
            last_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            last_hidden = hidden[-1]
        
        # Generate predictions
        predictions = []
        h = last_hidden.unsqueeze(1)
        c = cell[-1].unsqueeze(1) if not self.bidirectional else cell[-1].unsqueeze(1)
        
        for _ in range(self.prediction_length):
            # Use last hidden state to predict next step
            out = self.fc(self.dropout(h.squeeze(1)))
            predictions.append(out)
            
            # Use prediction as input for next step (teacher forcing disabled)
            h, c = self.lstm(out.unsqueeze(1), (h.transpose(0, 1), c.transpose(0, 1)))
            h = h.transpose(0, 1)
            c = c.transpose(0, 1)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=1)
        
        return predictions


class GRU(nn.Module):
    """GRU model for time series forecasting"""
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        prediction_length: int = 12,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        
        # GRU layers
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # GRU forward
        gru_out, hidden = self.gru(x)
        
        # Generate predictions
        predictions = []
        h = hidden[-1].unsqueeze(1)
        
        for _ in range(self.prediction_length):
            out = self.fc(self.dropout(h.squeeze(1)))
            predictions.append(out)
            
            # Use prediction as input for next step
            h, _ = self.gru(out.unsqueeze(1), h.transpose(0, 1))
            h = h.transpose(0, 1)
        
        predictions = torch.stack(predictions, dim=1)
        
        return predictions


class TransformerModel(nn.Module):
    """Transformer model for time series forecasting"""
    
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        prediction_length: int = 12,
        max_seq_length: int = 100
    ):
        super().__init__()
        
        self.d_model = d_model
        self.prediction_length = prediction_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
        
        # Decoder input embedding
        self.decoder_input_embedding = nn.Embedding(1, d_model)
    
    def forward(self, src):
        # Project input
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        
        # Create decoder input (learned embeddings)
        batch_size = src.size(0)
        tgt_input = torch.zeros(batch_size, self.prediction_length, 1).long().to(src.device)
        tgt = self.decoder_input_embedding(tgt_input).squeeze(2)
        tgt = self.pos_encoder(tgt)
        
        # Generate target mask
        tgt_mask = self.generate_square_subsequent_mask(self.prediction_length).to(src.device)
        
        # Transformer forward
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        
        # Project to output
        output = self.output_projection(output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


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


class Informer(nn.Module):
    """Informer model for long sequence time series forecasting"""
    
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        prediction_length: int = 12,
        factor: int = 5,
        distil: bool = True
    ):
        super().__init__()
        
        self.prediction_length = prediction_length
        self.d_model = d_model
        
        # Input embeddings
        self.enc_embedding = DataEmbedding(input_dim, d_model, dropout)
        self.dec_embedding = DataEmbedding(input_dim, d_model, dropout)
        
        # Encoder
        self.encoder = InformerEncoder(
            [
                InformerEncoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                    factor
                ) for _ in range(num_encoder_layers)
            ],
            distil
        )
        
        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                batch_first=True
            ),
            num_decoder_layers
        )
        
        # Output
        self.projection = nn.Linear(d_model, 1)
    
    def forward(self, x_enc, x_dec=None):
        # Encoder
        enc_out = self.enc_embedding(x_enc)
        enc_out = self.encoder(enc_out)
        
        # Decoder
        if x_dec is None:
            # Create decoder input
            batch_size = x_enc.size(0)
            x_dec = torch.zeros(batch_size, self.prediction_length, x_enc.size(2)).to(x_enc.device)
        
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out)
        
        # Project
        output = self.projection(dec_out)
        
        return output


class DataEmbedding(nn.Module):
    """Data embedding for Informer"""
    
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEncoding(d_model, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.value_embedding(x)
        x = self.position_embedding(x)
        return self.dropout(x)


class InformerEncoderLayer(nn.Module):
    """Informer encoder layer with ProbSparse attention"""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        factor: int = 5
    ):
        super().__init__()
        
        # ProbSparse self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + x
        
        return x


class InformerEncoder(nn.Module):
    """Informer encoder with distilling"""
    
    def __init__(self, layers: List[InformerEncoderLayer], distil: bool = True):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.distil = distil
        
        if distil:
            self.conv_layers = nn.ModuleList([
                nn.Conv1d(1, 1, kernel_size=3, padding=1, stride=2)
                for _ in range(len(layers) - 1)
            ])
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Distilling
            if self.distil and i < len(self.layers) - 1:
                x = x.transpose(1, 2)
                x = self.conv_layers[i](x)
                x = x.transpose(1, 2)
        
        return x


class TCN(nn.Module):
    """Temporal Convolutional Network"""
    
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        num_channels: List[int] = [64, 64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
        prediction_length: int = 12
    ):
        super().__init__()
        
        self.prediction_length = prediction_length
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_dim * prediction_length)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        
        y = self.network(x)
        
        # Global pooling
        y = F.adaptive_avg_pool1d(y, 1).squeeze(-1)
        
        # Project to predictions
        y = self.linear(y)
        y = y.view(-1, self.prediction_length, 1)
        
        return y


class TemporalBlock(nn.Module):
    """Temporal block for TCN"""
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove future time steps"""
    
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class NBeats(nn.Module):
    """N-BEATS model for time series forecasting"""
    
    def __init__(
        self,
        input_dim: int = 1,
        prediction_length: int = 12,
        stack_types: List[str] = ['trend', 'seasonality', 'generic'],
        num_blocks_per_stack: int = 3,
        hidden_dim: int = 256,
        share_weights: bool = False
    ):
        super().__init__()
        
        self.prediction_length = prediction_length
        self.stacks = nn.ModuleList()
        
        for stack_type in stack_types:
            blocks = []
            for _ in range(num_blocks_per_stack):
                if stack_type == 'trend':
                    block = TrendBlock(input_dim, prediction_length, hidden_dim)
                elif stack_type == 'seasonality':
                    block = SeasonalityBlock(input_dim, prediction_length, hidden_dim)
                else:
                    block = GenericBlock(input_dim, prediction_length, hidden_dim)
                
                blocks.append(block)
            
            self.stacks.append(nn.ModuleList(blocks))
    
    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        backcast_sum = 0
        forecast_sum = 0
        
        for stack in self.stacks:
            for block in stack:
                backcast, forecast = block(x)
                x = x - backcast
                backcast_sum = backcast_sum + backcast
                forecast_sum = forecast_sum + forecast
        
        return forecast_sum


class GenericBlock(nn.Module):
    """Generic block for N-BEATS"""
    
    def __init__(self, input_size: int, output_size: int, hidden_dim: int):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        
        self.backcast_fc = nn.Linear(hidden_dim, input_size)
        self.forecast_fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        # Flatten input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Forward through network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        # Generate backcast and forecast
        backcast = self.backcast_fc(x).view(batch_size, -1, 1)
        forecast = self.forecast_fc(x).view(batch_size, -1, 1)
        
        return backcast, forecast


class TrendBlock(GenericBlock):
    """Trend block for N-BEATS"""
    pass


class SeasonalityBlock(GenericBlock):
    """Seasonality block for N-BEATS"""
    pass


def get_model(
    model_name: str,
    input_dim: int = 1,
    prediction_length: int = 12,
    **kwargs
) -> nn.Module:
    """
    Factory function to get time series model
    
    Args:
        model_name: Name of the model
        input_dim: Input dimension
        prediction_length: Length of prediction horizon
        **kwargs: Additional model-specific arguments
    
    Returns:
        Time series model
    """
    model_name = model_name.lower()
    
    if model_name == 'lstm':
        return LSTM(input_dim=input_dim, prediction_length=prediction_length, **kwargs)
    elif model_name == 'gru':
        return GRU(input_dim=input_dim, prediction_length=prediction_length, **kwargs)
    elif model_name == 'transformer':
        return TransformerModel(input_dim=input_dim, prediction_length=prediction_length, **kwargs)
    elif model_name == 'informer':
        return Informer(input_dim=input_dim, prediction_length=prediction_length, **kwargs)
    elif model_name == 'tcn':
        return TCN(input_dim=input_dim, prediction_length=prediction_length, **kwargs)
    elif model_name == 'nbeats':
        return NBeats(input_dim=input_dim, prediction_length=prediction_length, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 32
    seq_len = 100
    input_dim = 3
    pred_len = 12
    
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    
    # Test LSTM
    model = LSTM(input_dim=input_dim, prediction_length=pred_len).to(device)
    output = model(x)
    print(f"LSTM output shape: {output.shape}")
    
    # Test Transformer
    model = TransformerModel(input_dim=input_dim, prediction_length=pred_len).to(device)
    output = model(x)
    print(f"Transformer output shape: {output.shape}")
    
    # Test TCN
    model = TCN(input_dim=input_dim, prediction_length=pred_len).to(device)
    output = model(x)
    print(f"TCN output shape: {output.shape}")