"""
Model architectures for machine translation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    MarianMTModel, MarianTokenizer,
    M2M100ForConditionalGeneration,
    MBartForConditionalGeneration,
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM
)
import math
from typing import Optional, Tuple, Dict, List


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerTranslator(nn.Module):
    """Transformer-based sequence-to-sequence model for translation"""
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Source embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.src_pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Target embeddings
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
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
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Source embedding
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.src_pos_encoding(src)
        src = self.dropout(src)
        
        # Target embedding
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.tgt_pos_encoding(tgt)
        tgt = self.dropout(tgt)
        
        # Generate target mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transformer
        output = self.transformer(
            src, tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Output projection
        output = self.output_projection(output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence"""
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.src_pos_encoding(src)
        src = self.dropout(src)
        
        memory = self.transformer.encoder(src, src_mask)
        return memory
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode target sequence"""
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.tgt_pos_encoding(tgt)
        tgt = self.dropout(tgt)
        
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        output = self.transformer.decoder(tgt, memory, tgt_mask)
        output = self.output_projection(output)
        
        return output


class Seq2SeqLSTM(nn.Module):
    """LSTM-based sequence-to-sequence model"""
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        
        # Encoder
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            embedding_dim,
            hidden_dim * 2,  # *2 for bidirectional encoder
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention
        if use_attention:
            self.attention = BahdanauAttention(hidden_dim * 2)
        
        # Output projection
        self.output_projection = nn.Linear(
            hidden_dim * 4 if use_attention else hidden_dim * 2,
            tgt_vocab_size
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Encode
        encoder_outputs, (hidden, cell) = self.encode(src, src_lengths)
        
        # Decode
        output = self.decode(tgt, hidden, cell, encoder_outputs)
        
        return output
    
    def encode(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode source sequence"""
        embedded = self.dropout(self.src_embedding(src))
        
        if src_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths, batch_first=True, enforce_sorted=False
            )
        
        outputs, (hidden, cell) = self.encoder(embedded)
        
        if src_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # Combine bidirectional hidden states
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)
        
        cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=-1)
        
        return outputs, (hidden, cell)
    
    def decode(
        self,
        tgt: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        """Decode target sequence"""
        embedded = self.dropout(self.tgt_embedding(tgt))
        
        output, (hidden, cell) = self.decoder(embedded, (hidden, cell))
        
        if self.use_attention:
            # Apply attention
            context = self.attention(output, encoder_outputs)
            output = torch.cat([output, context], dim=-1)
        
        output = self.output_projection(output)
        
        return output


class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        # decoder_hidden: (batch, seq_len, hidden_dim)
        # encoder_outputs: (batch, src_len, hidden_dim)
        
        # Calculate attention scores
        score = self.V(torch.tanh(
            self.W1(decoder_hidden).unsqueeze(2) + 
            self.W2(encoder_outputs).unsqueeze(1)
        )).squeeze(-1)
        
        # Apply softmax
        attention_weights = F.softmax(score, dim=-1)
        
        # Calculate context vector
        context = torch.bmm(attention_weights, encoder_outputs)
        
        return context


class MarianTranslator:
    """Wrapper for Marian MT models"""
    
    def __init__(self, model_name: str = 'Helsinki-NLP/opus-mt-en-de'):
        self.model = MarianMTModel.from_pretrained(model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    def translate(self, texts: List[str]) -> List[str]:
        """Translate texts"""
        # Tokenize
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        
        # Generate translations
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        
        # Decode
        translations = [self.tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
        
        return translations


def get_model(
    model_name: str,
    src_vocab_size: int = 32000,
    tgt_vocab_size: int = 32000,
    **kwargs
) -> nn.Module:
    """
    Factory function to get translation model
    
    Args:
        model_name: Name of the model
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        **kwargs: Additional model-specific arguments
    
    Returns:
        Translation model
    """
    model_name = model_name.lower()
    
    if model_name == 'transformer':
        return TransformerTranslator(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            **kwargs
        )
    elif model_name == 'lstm':
        return Seq2SeqLSTM(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            **kwargs
        )
    elif model_name == 'marian':
        model_path = kwargs.get('model_path', 'Helsinki-NLP/opus-mt-en-de')
        return MarianMTModel.from_pretrained(model_path)
    elif model_name == 'm2m100':
        return M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
    elif model_name == 'mbart':
        return MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')
    elif model_name == 't5':
        return T5ForConditionalGeneration.from_pretrained('t5-base')
    else:
        # Try to load from HuggingFace
        try:
            return AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except:
            raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test TransformerTranslator
    model = TransformerTranslator(
        src_vocab_size=10000,
        tgt_vocab_size=10000
    ).to(device)
    
    src = torch.randint(0, 10000, (2, 20)).to(device)
    tgt = torch.randint(0, 10000, (2, 15)).to(device)
    
    output = model(src, tgt)
    print(f"TransformerTranslator output shape: {output.shape}")
    
    # Test Seq2SeqLSTM
    model = Seq2SeqLSTM(
        src_vocab_size=10000,
        tgt_vocab_size=10000
    ).to(device)
    
    output = model(src, tgt)
    print(f"Seq2SeqLSTM output shape: {output.shape}")