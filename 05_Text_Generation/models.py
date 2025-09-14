"""
Model architectures for text generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    GPT2Model, GPT2LMHeadModel, GPT2Config,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    AutoModelForCausalLM
)
from typing import Optional, Dict, Tuple
import math


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
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerLM(nn.Module):
    """Custom Transformer Language Model"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        max_seq_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Dropout
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
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Token embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device), 
            diagonal=1
        ).bool()
        
        # Transformer encoding
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits


class GPT2FineTuner(nn.Module):
    """GPT-2 model for fine-tuning"""
    
    def __init__(
        self,
        model_name: str = 'gpt2',
        num_layers_to_freeze: int = 0,
        add_adapter: bool = False,
        adapter_size: int = 64
    ):
        super().__init__()
        
        # Load pretrained GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Freeze layers if specified
        if num_layers_to_freeze > 0:
            for param in self.gpt2.transformer.h[:num_layers_to_freeze].parameters():
                param.requires_grad = False
        
        # Add adapter layers if specified
        if add_adapter:
            self._add_adapters(adapter_size)
    
    def _add_adapters(self, adapter_size: int):
        """Add adapter layers to the model"""
        for layer in self.gpt2.transformer.h:
            # Add adapter after attention
            layer.attn_adapter = nn.Sequential(
                nn.Linear(layer.attn.c_proj.out_features, adapter_size),
                nn.ReLU(),
                nn.Linear(adapter_size, layer.attn.c_proj.out_features)
            )
            
            # Add adapter after FFN
            layer.ffn_adapter = nn.Sequential(
                nn.Linear(layer.mlp.c_proj.out_features, adapter_size),
                nn.ReLU(),
                nn.Linear(adapter_size, layer.mlp.c_proj.out_features)
            )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict:
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs


class LSTMGenerator(nn.Module):
    """LSTM-based text generator"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Embeddings
        embedded = self.embedding(input_ids)
        
        # LSTM forward
        output, hidden = self.lstm(embedded, hidden)
        
        # Output projection
        logits = self.output(output)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


class ConditionalGenerator(nn.Module):
    """Conditional text generator with control codes"""
    
    def __init__(
        self,
        vocab_size: int,
        num_conditions: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.condition_embedding = nn.Embedding(num_conditions, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        condition_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get embeddings
        token_emb = self.token_embedding(input_ids)
        condition_emb = self.condition_embedding(condition_ids).unsqueeze(1)
        
        # Concatenate condition embedding at the beginning
        x = torch.cat([condition_emb, token_emb], dim=1)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()
        
        # Transformer encoding
        x = self.transformer(x, mask=causal_mask)
        
        # Output projection (skip the condition token)
        logits = self.output_projection(x[:, 1:, :])
        
        return logits


def get_model(
    model_name: str,
    vocab_size: int = 50257,  # GPT-2 vocab size
    **kwargs
) -> nn.Module:
    """
    Factory function to get text generation model
    
    Args:
        model_name: Name of the model
        vocab_size: Size of vocabulary
        **kwargs: Additional model-specific arguments
    
    Returns:
        Text generation model
    """
    model_name = model_name.lower()
    
    if model_name == 'gpt2':
        return GPT2LMHeadModel.from_pretrained('gpt2')
    elif model_name == 'gpt2-medium':
        return GPT2LMHeadModel.from_pretrained('gpt2-medium')
    elif model_name == 'gpt2-large':
        return GPT2LMHeadModel.from_pretrained('gpt2-large')
    elif model_name == 'gpt2-finetuner':
        return GPT2FineTuner(**kwargs)
    elif model_name == 't5':
        return T5ForConditionalGeneration.from_pretrained('t5-small')
    elif model_name == 'bart':
        return BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    elif model_name == 'transformer':
        return TransformerLM(vocab_size=vocab_size, **kwargs)
    elif model_name == 'lstm':
        return LSTMGenerator(vocab_size=vocab_size, **kwargs)
    elif model_name == 'conditional':
        return ConditionalGenerator(vocab_size=vocab_size, **kwargs)
    else:
        # Try to load from HuggingFace
        try:
            return AutoModelForCausalLM.from_pretrained(model_name)
        except:
            raise ValueError(f"Unknown model: {model_name}")


class TextGenerator:
    """Utility class for text generation"""
    
    def __init__(self, model: nn.Module, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> List[str]:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            num_return_sequences: Number of sequences to generate
            do_sample: Whether to use sampling
        
        Returns:
            List of generated texts
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test TransformerLM
    model = TransformerLM(vocab_size=10000).to(device)
    input_ids = torch.randint(0, 10000, (2, 50)).to(device)
    logits = model(input_ids)
    print(f"TransformerLM output shape: {logits.shape}")
    
    # Test LSTMGenerator
    model = LSTMGenerator(vocab_size=10000).to(device)
    hidden = model.init_hidden(2, device)
    logits, hidden = model(input_ids, hidden)
    print(f"LSTMGenerator output shape: {logits.shape}")
    
    # Test ConditionalGenerator
    model = ConditionalGenerator(vocab_size=10000, num_conditions=5).to(device)
    condition_ids = torch.randint(0, 5, (2,)).to(device)
    logits = model(input_ids, condition_ids)
    print(f"ConditionalGenerator output shape: {logits.shape}")