"""
Model architectures for Automatic Speech Recognition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math
from transformers import (
    Wav2Vec2ForCTC,
    WhisperForConditionalGeneration,
    AutoModelForSpeechSeq2Seq
)


class ConvBlock(nn.Module):
    """Convolutional block for feature extraction"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class DeepSpeech2(nn.Module):
    """DeepSpeech2-like architecture"""
    
    def __init__(
        self,
        input_dim: int = 161,  # Spectrogram features
        hidden_dim: int = 512,
        num_classes: int = 29,  # Alphabet + special tokens
        num_rnn_layers: int = 5,
        rnn_type: str = 'gru',
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            ConvBlock(input_dim, 32, kernel_size=41, stride=2, dropout=dropout),
            ConvBlock(32, 64, kernel_size=21, stride=2, dropout=dropout),
            ConvBlock(64, 128, kernel_size=11, stride=1, dropout=dropout)
        )
        
        # Calculate RNN input size
        rnn_input_size = 128
        
        # RNN layers
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                rnn_input_size,
                hidden_dim,
                num_rnn_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_rnn_layers > 1 else 0
            )
        else:
            self.rnn = nn.GRU(
                rnn_input_size,
                hidden_dim,
                num_rnn_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_rnn_layers > 1 else 0
            )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x, input_lengths=None):
        # x shape: (batch, time, features) for spectrograms
        
        # Transpose for conv layers (batch, features, time)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Transpose back for RNN (batch, time, features)
        x = x.transpose(1, 2)
        
        # RNN layers
        if input_lengths is not None:
            # Pack padded sequence
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True, enforce_sorted=False
            )
            x, _ = self.rnn(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        else:
            x, _ = self.rnn(x)
        
        # Output projection
        x = self.fc(x)
        
        # Log softmax for CTC loss
        x = F.log_softmax(x, dim=-1)
        
        return x


class ConformerBlock(nn.Module):
    """Conformer block combining convolution and self-attention"""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Feed-forward module 1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Convolution module
        self.conv_module = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Conv1d(
                d_model,
                d_model * 2,
                kernel_size=1
            ),
            nn.GLU(dim=1),
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=d_model
            ),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # Feed-forward module 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Feed-forward 1
        residual = x
        x = self.ff1(x)
        x = residual + 0.5 * x
        
        # Self-attention
        residual = x
        x = self.attn_norm(x)
        x, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.attn_dropout(x)
        x = residual + x
        
        # Convolution
        residual = x
        x_conv = x.transpose(1, 2)  # (batch, channels, time)
        x_conv = self.conv_module(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (batch, time, channels)
        x = residual + x_conv
        
        # Feed-forward 2
        residual = x
        x = self.ff2(x)
        x = residual + 0.5 * x
        
        # Final layer norm
        x = self.layer_norm(x)
        
        return x


class Conformer(nn.Module):
    """Conformer model for ASR"""
    
    def __init__(
        self,
        input_dim: int = 80,  # Mel-spectrogram features
        d_model: int = 256,
        num_layers: int = 12,
        nhead: int = 4,
        conv_kernel_size: int = 31,
        num_classes: int = 29,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, nhead, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, num_classes)
    
    def forward(self, x, mask=None):
        # Input projection
        x = self.input_projection(x)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Conformer blocks
        for block in self.conformer_blocks:
            x = block(x, mask)
        
        # Output projection
        x = self.output_projection(x)
        
        # Log softmax for CTC loss
        x = F.log_softmax(x, dim=-1)
        
        return x


class LAS(nn.Module):
    """Listen, Attend and Spell (LAS) model"""
    
    def __init__(
        self,
        input_dim: int = 80,
        encoder_hidden: int = 256,
        decoder_hidden: int = 256,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        num_classes: int = 29,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Encoder (Listener)
        self.encoder = nn.LSTM(
            input_dim,
            encoder_hidden,
            num_encoder_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_encoder_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = BahdanauAttention(
            encoder_hidden * 2,
            decoder_hidden
        )
        
        # Decoder (Speller)
        self.decoder_cell = nn.LSTMCell(
            num_classes + encoder_hidden * 2,  # Input + context
            decoder_hidden
        )
        
        # Output projection
        self.output_projection = nn.Linear(
            decoder_hidden + encoder_hidden * 2,
            num_classes
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, targets=None, target_lengths=None):
        # Encode
        encoder_outputs, _ = self.encoder(x)
        encoder_outputs = self.dropout(encoder_outputs)
        
        batch_size = x.size(0)
        max_length = targets.size(1) if targets is not None else 100
        
        # Initialize decoder
        decoder_hidden = encoder_outputs.new_zeros(batch_size, self.decoder_cell.hidden_size)
        decoder_cell = encoder_outputs.new_zeros(batch_size, self.decoder_cell.hidden_size)
        
        outputs = []
        
        # Decoder loop
        for t in range(max_length):
            if targets is not None and t > 0:
                # Teacher forcing
                input_t = targets[:, t-1]
            else:
                # Use previous prediction
                if t == 0:
                    input_t = torch.zeros(batch_size, dtype=torch.long).to(x.device)
                else:
                    input_t = outputs[-1].argmax(dim=-1)
            
            # Embed input
            input_embedded = F.one_hot(input_t, num_classes=self.output_projection.out_features).float()
            
            # Attention
            context = self.attention(decoder_hidden, encoder_outputs)
            
            # Decoder step
            decoder_input = torch.cat([input_embedded, context], dim=-1)
            decoder_hidden, decoder_cell = self.decoder_cell(
                decoder_input,
                (decoder_hidden, decoder_cell)
            )
            
            # Output
            output = torch.cat([decoder_hidden, context], dim=-1)
            output = self.output_projection(output)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs


class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism"""
    
    def __init__(self, encoder_hidden: int, decoder_hidden: int):
        super().__init__()
        
        self.W1 = nn.Linear(encoder_hidden, decoder_hidden, bias=False)
        self.W2 = nn.Linear(decoder_hidden, decoder_hidden, bias=False)
        self.V = nn.Linear(decoder_hidden, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, decoder_hidden)
        # encoder_outputs: (batch, seq_len, encoder_hidden)
        
        # Calculate attention scores
        scores = self.V(torch.tanh(
            self.W1(encoder_outputs) + 
            self.W2(decoder_hidden).unsqueeze(1)
        )).squeeze(-1)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Calculate context vector
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)
        
        return context


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
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


class TransformerASR(nn.Module):
    """Transformer-based ASR model"""
    
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_classes: int = 29,
        dropout: float = 0.1,
        max_seq_length: int = 1000
    ):
        super().__init__()
        
        self.d_model = d_model
        
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
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, num_classes)
        
        # Target embedding
        self.tgt_embedding = nn.Embedding(num_classes, d_model)
    
    def forward(self, src, tgt=None):
        # Source encoding
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        
        if tgt is not None:
            # Target embedding
            tgt = self.tgt_embedding(tgt)
            tgt = self.pos_encoder(tgt)
            
            # Generate target mask
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
            # Transformer
            output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        else:
            # Encoder only (for CTC)
            encoder = self.transformer.encoder
            output = encoder(src)
        
        # Output projection
        output = self.output_projection(output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


def get_model(
    model_name: str,
    input_dim: int = 80,
    num_classes: int = 29,
    **kwargs
) -> nn.Module:
    """
    Factory function to get ASR model
    
    Args:
        model_name: Name of the model
        input_dim: Input dimension
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments
    
    Returns:
        ASR model
    """
    model_name = model_name.lower()
    
    if model_name == 'deepspeech2':
        return DeepSpeech2(
            input_dim=input_dim,
            num_classes=num_classes,
            **kwargs
        )
    elif model_name == 'conformer':
        return Conformer(
            input_dim=input_dim,
            num_classes=num_classes,
            **kwargs
        )
    elif model_name == 'las':
        return LAS(
            input_dim=input_dim,
            num_classes=num_classes,
            **kwargs
        )
    elif model_name == 'transformer':
        return TransformerASR(
            input_dim=input_dim,
            num_classes=num_classes,
            **kwargs
        )
    elif model_name == 'wav2vec2':
        return Wav2Vec2ForCTC.from_pretrained(
            kwargs.get('pretrained_model', 'facebook/wav2vec2-base-960h')
        )
    elif model_name == 'whisper':
        return WhisperForConditionalGeneration.from_pretrained(
            kwargs.get('pretrained_model', 'openai/whisper-base')
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test DeepSpeech2
    model = DeepSpeech2(input_dim=161, num_classes=29).to(device)
    x = torch.randn(2, 100, 161).to(device)  # (batch, time, features)
    output = model(x)
    print(f"DeepSpeech2 output shape: {output.shape}")
    
    # Test Conformer
    model = Conformer(input_dim=80, num_classes=29).to(device)
    x = torch.randn(2, 100, 80).to(device)
    output = model(x)
    print(f"Conformer output shape: {output.shape}")
    
    # Test LAS
    model = LAS(input_dim=80, num_classes=29).to(device)
    x = torch.randn(2, 100, 80).to(device)
    targets = torch.randint(0, 29, (2, 20)).to(device)
    output = model(x, targets)
    print(f"LAS output shape: {output.shape}")
    
    # Test TransformerASR
    model = TransformerASR(input_dim=80, num_classes=29).to(device)
    x = torch.randn(2, 100, 80).to(device)
    output = model(x)
    print(f"TransformerASR output shape: {output.shape}")