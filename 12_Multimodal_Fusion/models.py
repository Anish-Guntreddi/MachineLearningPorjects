"""
Model architectures for Multimodal Fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math
from transformers import AutoModel


class MultimodalFusion(nn.Module):
    """Base class for multimodal fusion models"""
    
    def __init__(
        self,
        image_dim: int = 2048,
        text_dim: int = 768,
        audio_dim: int = 128,
        fusion_dim: int = 512,
        output_dim: int = 10,
        fusion_type: str = 'concat',
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.modalities = []
        
        # Modality-specific encoders
        self.image_encoder = None
        self.text_encoder = None
        self.audio_encoder = None
        
        # Fusion layer
        if fusion_type == 'concat':
            total_dim = 0
            if image_dim > 0:
                total_dim += fusion_dim
                self.modalities.append('image')
            if text_dim > 0:
                total_dim += fusion_dim
                self.modalities.append('text')
            if audio_dim > 0:
                total_dim += fusion_dim
                self.modalities.append('audio')
            
            self.fusion_layer = nn.Sequential(
                nn.Linear(total_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, output_dim)
            )
        elif fusion_type == 'attention':
            self.fusion_layer = CrossModalAttention(
                fusion_dim, num_heads=8, dropout=dropout
            )
            self.output_layer = nn.Linear(fusion_dim, output_dim)


class EarlyFusion(MultimodalFusion):
    """Early fusion: Combine features before processing"""
    
    def __init__(
        self,
        image_dim: int = 2048,
        text_dim: int = 768,
        audio_dim: int = 128,
        fusion_dim: int = 512,
        output_dim: int = 10,
        dropout: float = 0.2
    ):
        super().__init__(
            image_dim, text_dim, audio_dim,
            fusion_dim, output_dim, 'concat', dropout
        )
        
        # Project each modality to same dimension
        if image_dim > 0:
            self.image_proj = nn.Linear(image_dim, fusion_dim)
        if text_dim > 0:
            self.text_proj = nn.Linear(text_dim, fusion_dim)
        if audio_dim > 0:
            self.audio_proj = nn.Linear(audio_dim, fusion_dim)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        
        if 'image' in inputs and hasattr(self, 'image_proj'):
            image_feat = self.image_proj(inputs['image'])
            features.append(image_feat)
        
        if 'text' in inputs and hasattr(self, 'text_proj'):
            text_feat = self.text_proj(inputs['text'])
            features.append(text_feat)
        
        if 'audio' in inputs and hasattr(self, 'audio_proj'):
            audio_feat = self.audio_proj(inputs['audio'])
            features.append(audio_feat)
        
        # Concatenate features
        fused = torch.cat(features, dim=-1)
        output = self.fusion_layer(fused)
        
        return output


class LateFusion(MultimodalFusion):
    """Late fusion: Process each modality separately then combine"""
    
    def __init__(
        self,
        image_dim: int = 2048,
        text_dim: int = 768,
        audio_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 10,
        dropout: float = 0.2
    ):
        super().__init__(
            image_dim, text_dim, audio_dim,
            hidden_dim, output_dim, 'late', dropout
        )
        
        # Separate networks for each modality
        if image_dim > 0:
            self.image_network = nn.Sequential(
                nn.Linear(image_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        
        if text_dim > 0:
            self.text_network = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        
        if audio_dim > 0:
            self.audio_network = nn.Sequential(
                nn.Linear(audio_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = []
        
        if 'image' in inputs and hasattr(self, 'image_network'):
            image_out = self.image_network(inputs['image'])
            outputs.append(image_out)
        
        if 'text' in inputs and hasattr(self, 'text_network'):
            text_out = self.text_network(inputs['text'])
            outputs.append(text_out)
        
        if 'audio' in inputs and hasattr(self, 'audio_network'):
            audio_out = self.audio_network(inputs['audio'])
            outputs.append(audio_out)
        
        # Average predictions
        output = torch.stack(outputs).mean(dim=0)
        
        return output


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim
        )
        
        # Output projection
        output = self.out_linear(context)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        return output


class HierarchicalFusion(nn.Module):
    """Hierarchical fusion with multiple levels"""
    
    def __init__(
        self,
        image_dim: int = 2048,
        text_dim: int = 768,
        audio_dim: int = 128,
        hidden_dims: List[int] = [512, 256, 128],
        output_dim: int = 10,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # First level: pairwise fusion
        self.image_text_fusion = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.image_audio_fusion = nn.Sequential(
            nn.Linear(image_dim + audio_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.text_audio_fusion = nn.Sequential(
            nn.Linear(text_dim + audio_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Second level: fusion of pairwise features
        self.second_fusion = nn.Sequential(
            nn.Linear(hidden_dims[0] * 3, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[2], output_dim)
        )
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        image = inputs.get('image')
        text = inputs.get('text')
        audio = inputs.get('audio')
        
        # First level fusion
        if image is not None and text is not None:
            image_text = self.image_text_fusion(torch.cat([image, text], dim=-1))
        else:
            image_text = torch.zeros(image.shape[0], 512).to(image.device)
        
        if image is not None and audio is not None:
            image_audio = self.image_audio_fusion(torch.cat([image, audio], dim=-1))
        else:
            image_audio = torch.zeros(image.shape[0], 512).to(image.device)
        
        if text is not None and audio is not None:
            text_audio = self.text_audio_fusion(torch.cat([text, audio], dim=-1))
        else:
            text_audio = torch.zeros(text.shape[0], 512).to(text.device)
        
        # Second level fusion
        combined = torch.cat([image_text, image_audio, text_audio], dim=-1)
        fused = self.second_fusion(combined)
        
        # Output
        output = self.output_layer(fused)
        
        return output


class VQAModel(nn.Module):
    """Visual Question Answering model"""
    
    def __init__(
        self,
        image_encoder: str = 'resnet',
        text_encoder: str = 'bert',
        image_dim: int = 2048,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_answers: int = 1000,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Image encoder
        if image_encoder == 'resnet':
            self.image_encoder = nn.Sequential(
                nn.Linear(image_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Text encoder
        if text_encoder == 'bert':
            self.text_encoder = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(
            hidden_dim, num_heads=8, dropout=dropout
        )
        
        # Answer prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_answers)
        )
    
    def forward(
        self,
        image: torch.Tensor,
        question: torch.Tensor,
        question_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Encode image and question
        image_feat = self.image_encoder(image)
        question_feat = self.text_encoder(question)
        
        # Cross-modal attention
        # Image attends to question
        image_attended = self.cross_attention(
            image_feat.unsqueeze(1),
            question_feat.unsqueeze(1),
            question_feat.unsqueeze(1)
        ).squeeze(1)
        
        # Question attends to image
        question_attended = self.cross_attention(
            question_feat.unsqueeze(1),
            image_feat.unsqueeze(1),
            image_feat.unsqueeze(1)
        ).squeeze(1)
        
        # Combine features
        combined = torch.cat([image_attended, question_attended], dim=-1)
        
        # Predict answer
        logits = self.classifier(combined)
        
        return logits


class CLIPModel(nn.Module):
    """CLIP-style image-text matching model"""
    
    def __init__(
        self,
        image_dim: int = 2048,
        text_dim: int = 768,
        embed_dim: int = 512,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.temperature = temperature
        
        # Image encoder
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        image_features = F.normalize(image_features, dim=-1)
        return image_features
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        text_features = self.text_encoder(text)
        text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # Cosine similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text


class MultimodalTransformer(nn.Module):
    """Transformer-based multimodal fusion"""
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_modalities: int = 3,
        output_dim: int = 10
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Modality embeddings
        self.modality_embeddings = nn.Embedding(num_modalities, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, output_dim)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = next(iter(inputs.values())).size(0)
        sequences = []
        
        # Process each modality
        for i, (modality, features) in enumerate(inputs.items()):
            # Add modality embedding
            mod_emb = self.modality_embeddings(
                torch.tensor([i]).to(features.device)
            ).expand(batch_size, 1, -1)
            
            # Ensure features have sequence dimension
            if len(features.shape) == 2:
                features = features.unsqueeze(1)
            
            # Project to d_model if needed
            if features.size(-1) != self.d_model:
                proj = nn.Linear(features.size(-1), self.d_model).to(features.device)
                features = proj(features)
            
            features = features + mod_emb
            sequences.append(features)
        
        # Concatenate all modalities
        sequence = torch.cat(sequences, dim=1)
        
        # Apply positional encoding
        sequence = self.pos_encoder(sequence)
        
        # Transformer encoding
        sequence = sequence.transpose(0, 1)  # (seq_len, batch, d_model)
        encoded = self.transformer_encoder(sequence)
        
        # Global pooling
        pooled = encoded.mean(dim=0)
        
        # Output
        output = self.output_layer(pooled)
        
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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class GatedFusion(nn.Module):
    """Gated multimodal fusion"""
    
    def __init__(
        self,
        input_dims: List[int],
        hidden_dim: int = 256,
        output_dim: int = 10,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_modalities = len(input_dims)
        
        # Transform each modality
        self.transforms = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])
        
        # Gating network
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            ) for dim in input_dims
        ])
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Transform and gate each modality
        gated_features = []
        
        for i, x in enumerate(inputs):
            transformed = self.transforms[i](x)
            gate = self.gates[i](x)
            gated = transformed * gate
            gated_features.append(gated)
        
        # Sum gated features
        fused = sum(gated_features)
        
        # Output
        output = self.output_layer(fused)
        
        return output


class TensorFusion(nn.Module):
    """Tensor fusion network"""
    
    def __init__(
        self,
        input_dims: List[int],
        output_dim: int = 10,
        rank: int = 10,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dims = input_dims
        self.rank = rank
        
        # Low-rank factors for each modality
        self.factors = nn.ParameterList([
            nn.Parameter(torch.randn(dim + 1, rank)) for dim in input_dims
        ])
        
        # Output projection
        fusion_dim = rank ** len(input_dims)
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        batch_size = inputs[0].size(0)
        
        # Add bias term to each input
        inputs_with_bias = []
        for x in inputs:
            bias = torch.ones(batch_size, 1).to(x.device)
            x_bias = torch.cat([x, bias], dim=1)
            inputs_with_bias.append(x_bias)
        
        # Compute tensor product using low-rank approximation
        fusion = None
        for i, (x, factor) in enumerate(zip(inputs_with_bias, self.factors)):
            projected = x @ factor  # (batch, rank)
            
            if fusion is None:
                fusion = projected
            else:
                # Outer product
                fusion = fusion.unsqueeze(-1) @ projected.unsqueeze(1)
                fusion = fusion.view(batch_size, -1)
        
        # Output
        output = self.output_layer(fusion)
        
        return output


def get_fusion_model(
    model_name: str,
    **kwargs
) -> nn.Module:
    """
    Factory function to get multimodal fusion model
    
    Args:
        model_name: Name of the model
        **kwargs: Model-specific arguments
    
    Returns:
        Multimodal fusion model
    """
    model_name = model_name.lower()
    
    if model_name == 'early_fusion':
        return EarlyFusion(**kwargs)
    elif model_name == 'late_fusion':
        return LateFusion(**kwargs)
    elif model_name == 'hierarchical':
        return HierarchicalFusion(**kwargs)
    elif model_name == 'vqa':
        return VQAModel(**kwargs)
    elif model_name == 'clip':
        return CLIPModel(**kwargs)
    elif model_name == 'transformer':
        return MultimodalTransformer(**kwargs)
    elif model_name == 'gated':
        return GatedFusion(**kwargs)
    elif model_name == 'tensor':
        return TensorFusion(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 16
    image_dim = 2048
    text_dim = 768
    audio_dim = 128
    
    # Test early fusion
    model = EarlyFusion(
        image_dim=image_dim,
        text_dim=text_dim,
        audio_dim=audio_dim
    ).to(device)
    
    inputs = {
        'image': torch.randn(batch_size, image_dim).to(device),
        'text': torch.randn(batch_size, text_dim).to(device),
        'audio': torch.randn(batch_size, audio_dim).to(device)
    }
    
    output = model(inputs)
    print(f"Early Fusion output shape: {output.shape}")
    
    # Test VQA model
    vqa_model = VQAModel(
        image_dim=image_dim,
        text_dim=text_dim
    ).to(device)
    
    image = torch.randn(batch_size, image_dim).to(device)
    question = torch.randn(batch_size, text_dim).to(device)
    
    vqa_output = vqa_model(image, question)
    print(f"VQA output shape: {vqa_output.shape}")
    
    # Test CLIP model
    clip_model = CLIPModel(
        image_dim=image_dim,
        text_dim=text_dim
    ).to(device)
    
    logits_i2t, logits_t2i = clip_model(image, question)
    print(f"CLIP logits shape: {logits_i2t.shape}")
    
    # Test Multimodal Transformer
    transformer_model = MultimodalTransformer(
        d_model=512,
        num_modalities=3,
        output_dim=10
    ).to(device)
    
    trans_output = transformer_model(inputs)
    print(f"Transformer output shape: {trans_output.shape}")