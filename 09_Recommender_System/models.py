"""
Model architectures for Recommender Systems
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np
import math


class MatrixFactorization(nn.Module):
    """Basic Matrix Factorization model"""
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 50,
        use_bias: bool = True
    ):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.use_bias = use_bias
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Biases
        if use_bias:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        if self.use_bias:
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, user_ids, item_ids):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Dot product
        output = (user_emb * item_emb).sum(dim=1)
        
        # Add biases
        if self.use_bias:
            user_b = self.user_bias(user_ids).squeeze()
            item_b = self.item_bias(item_ids).squeeze()
            output = output + user_b + item_b + self.global_bias
        
        return output


class NeuralCollaborativeFiltering(nn.Module):
    """Neural Collaborative Filtering (NCF) model"""
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        mf_dim: int = 8,
        mlp_dims: List[int] = [64, 32, 16, 8],
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.mf_dim = mf_dim
        self.mlp_dims = mlp_dims
        
        # GMF embeddings
        self.gmf_user_embedding = nn.Embedding(num_users, mf_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, mf_dim)
        
        # MLP embeddings
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_dims[0] // 2)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_dims[0] // 2)
        
        # MLP layers
        mlp_layers = []
        for i in range(len(mlp_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Prediction layer
        self.prediction = nn.Linear(mf_dim + mlp_dims[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.gmf_user_embedding.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        
        nn.init.xavier_uniform_(self.prediction.weight)
    
    def forward(self, user_ids, item_ids):
        # GMF part
        gmf_user = self.gmf_user_embedding(user_ids)
        gmf_item = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user * gmf_item
        
        # MLP part
        mlp_user = self.mlp_user_embedding(user_ids)
        mlp_item = self.mlp_item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=1)
        mlp_output = self.mlp(mlp_input)
        
        # Concatenate and predict
        concat = torch.cat([gmf_output, mlp_output], dim=1)
        output = self.prediction(concat)
        
        return output.squeeze()


class DeepFM(nn.Module):
    """Deep Factorization Machine model"""
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_features: int = 0,
        embedding_dim: int = 8,
        mlp_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        # First-order embeddings
        self.user_embedding_1st = nn.Embedding(num_users, 1)
        self.item_embedding_1st = nn.Embedding(num_items, 1)
        
        # Second-order embeddings
        self.user_embedding_2nd = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_2nd = nn.Embedding(num_items, embedding_dim)
        
        # Feature embeddings if available
        if num_features > 0:
            self.feature_embedding_1st = nn.Embedding(num_features, 1)
            self.feature_embedding_2nd = nn.Embedding(num_features, embedding_dim)
        
        # Deep network
        input_dim = embedding_dim * 2  # User + Item
        if num_features > 0:
            input_dim += embedding_dim
        
        layers = []
        prev_dim = input_dim
        for dim in mlp_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.deep_network = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(mlp_dims[-1] + 1, 1)  # +1 for FM output
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for embedding in [self.user_embedding_1st, self.item_embedding_1st,
                         self.user_embedding_2nd, self.item_embedding_2nd]:
            nn.init.normal_(embedding.weight, std=0.01)
        
        if self.num_features > 0:
            nn.init.normal_(self.feature_embedding_1st.weight, std=0.01)
            nn.init.normal_(self.feature_embedding_2nd.weight, std=0.01)
    
    def forward(self, user_ids, item_ids, feature_ids=None):
        # First-order (linear) part
        linear_user = self.user_embedding_1st(user_ids)
        linear_item = self.item_embedding_1st(item_ids)
        linear_output = linear_user + linear_item
        
        if feature_ids is not None and self.num_features > 0:
            linear_feature = self.feature_embedding_1st(feature_ids).sum(dim=1)
            linear_output = linear_output + linear_feature
        
        # Second-order (FM) part
        fm_user = self.user_embedding_2nd(user_ids)
        fm_item = self.item_embedding_2nd(item_ids)
        
        # FM interaction
        fm_sum_square = (fm_user + fm_item) ** 2
        fm_square_sum = fm_user ** 2 + fm_item ** 2
        fm_output = 0.5 * (fm_sum_square - fm_square_sum).sum(dim=1, keepdim=True)
        
        # Deep part
        deep_input = torch.cat([fm_user, fm_item], dim=1)
        if feature_ids is not None and self.num_features > 0:
            fm_feature = self.feature_embedding_2nd(feature_ids).mean(dim=1)
            deep_input = torch.cat([deep_input, fm_feature], dim=1)
        
        deep_output = self.deep_network(deep_input)
        
        # Combine all parts
        concat = torch.cat([linear_output + fm_output, deep_output], dim=1)
        output = self.output_layer(concat)
        
        return output.squeeze()


class AutoEncoder(nn.Module):
    """AutoEncoder for collaborative filtering"""
    
    def __init__(
        self,
        num_items: int,
        encoder_dims: List[int] = [600, 200],
        decoder_dims: Optional[List[int]] = None,
        dropout: float = 0.5,
        activation: str = 'selu'
    ):
        super().__init__()
        
        self.num_items = num_items
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims or encoder_dims[::-1][1:]
        
        # Choose activation
        if activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
        
        # Encoder
        encoder_layers = []
        prev_dim = num_items
        for dim in encoder_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(self.activation)
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        for dim in self.decoder_dims:
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(self.activation)
            prev_dim = dim
        
        decoder_layers.append(nn.Linear(prev_dim, num_items))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        return decoded
    
    def encode(self, x):
        """Get latent representation"""
        return self.encoder(x)


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation"""
    
    def __init__(
        self,
        num_items: int,
        max_seq_length: int = 50,
        hidden_dim: int = 50,
        num_heads: int = 1,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_items = num_items
        self.max_seq_length = max_seq_length
        self.hidden_dim = hidden_dim
        
        # Item embedding
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        
        # Positional encoding
        self.pos_embedding = nn.Embedding(max_seq_length, hidden_dim)
        
        # Self-attention blocks
        self.attention_blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_items)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
    
    def forward(self, item_seq, seq_len=None):
        # Get embeddings
        batch_size, seq_length = item_seq.shape
        
        # Item embeddings
        item_emb = self.item_embedding(item_seq)
        
        # Position embeddings
        positions = torch.arange(seq_length, device=item_seq.device).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        
        # Combine embeddings
        seq_emb = item_emb + pos_emb
        seq_emb = self.layer_norm(seq_emb)
        seq_emb = self.dropout(seq_emb)
        
        # Create attention mask
        if seq_len is not None:
            mask = self._create_mask(seq_len, seq_length).to(item_seq.device)
        else:
            mask = None
        
        # Self-attention blocks
        for block in self.attention_blocks:
            seq_emb = block(seq_emb, mask)
        
        # Output scores for all items
        output = self.output_layer(seq_emb)
        
        return output
    
    def _create_mask(self, seq_len, max_len):
        """Create attention mask"""
        batch_size = len(seq_len)
        mask = torch.zeros(batch_size, max_len, max_len)
        
        for i, length in enumerate(seq_len):
            mask[i, :length, :length] = 1
        
        return mask.bool()


class SelfAttentionBlock(nn.Module):
    """Self-attention block for SASRec"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        
        return x


class LightGCN(nn.Module):
    """Light Graph Convolution Network for recommendation"""
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        alpha: Optional[float] = None
    ):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Layer combination weights
        if alpha is None:
            self.alpha = 1 / (num_layers + 1)
        else:
            self.alpha = alpha
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(self, user_ids, item_ids, edge_index=None):
        if edge_index is None:
            # Simple dot product without graph convolution
            user_emb = self.user_embedding(user_ids)
            item_emb = self.item_embedding(item_ids)
            return (user_emb * item_emb).sum(dim=1)
        
        # Graph convolution
        all_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ])
        
        embeddings_list = [all_embeddings]
        
        for layer in range(self.num_layers):
            # Simple graph convolution
            all_embeddings = self._graph_conv(all_embeddings, edge_index)
            embeddings_list.append(all_embeddings)
        
        # Layer combination
        final_embeddings = torch.stack(embeddings_list, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        
        # Split back to users and items
        user_final, item_final = torch.split(
            final_embeddings,
            [self.num_users, self.num_items]
        )
        
        # Get specific user and item embeddings
        user_emb = user_final[user_ids]
        item_emb = item_final[item_ids]
        
        # Dot product
        return (user_emb * item_emb).sum(dim=1)
    
    def _graph_conv(self, embeddings, edge_index):
        """Simple graph convolution"""
        # This is a simplified version
        # In practice, you'd use a proper graph library like PyTorch Geometric
        return embeddings


def get_model(
    model_name: str,
    num_users: int,
    num_items: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to get recommender model
    
    Args:
        model_name: Name of the model
        num_users: Number of users
        num_items: Number of items
        **kwargs: Additional model-specific arguments
    
    Returns:
        Recommender model
    """
    model_name = model_name.lower()
    
    if model_name == 'mf' or model_name == 'matrix_factorization':
        return MatrixFactorization(num_users, num_items, **kwargs)
    elif model_name == 'ncf':
        return NeuralCollaborativeFiltering(num_users, num_items, **kwargs)
    elif model_name == 'deepfm':
        return DeepFM(num_users, num_items, **kwargs)
    elif model_name == 'autoencoder':
        return AutoEncoder(num_items, **kwargs)
    elif model_name == 'sasrec':
        return SASRec(num_items, **kwargs)
    elif model_name == 'lightgcn':
        return LightGCN(num_users, num_items, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_users = 1000
    num_items = 500
    batch_size = 32
    
    # Test Matrix Factorization
    model = MatrixFactorization(num_users, num_items).to(device)
    users = torch.randint(0, num_users, (batch_size,)).to(device)
    items = torch.randint(0, num_items, (batch_size,)).to(device)
    output = model(users, items)
    print(f"MF output shape: {output.shape}")
    
    # Test NCF
    model = NeuralCollaborativeFiltering(num_users, num_items).to(device)
    output = model(users, items)
    print(f"NCF output shape: {output.shape}")
    
    # Test DeepFM
    model = DeepFM(num_users, num_items).to(device)
    output = model(users, items)
    print(f"DeepFM output shape: {output.shape}")
    
    # Test AutoEncoder
    model = AutoEncoder(num_items).to(device)
    user_items = torch.randn(batch_size, num_items).to(device)
    output = model(user_items)
    print(f"AutoEncoder output shape: {output.shape}")
    
    # Test SASRec
    model = SASRec(num_items).to(device)
    item_seq = torch.randint(0, num_items, (batch_size, 20)).to(device)
    output = model(item_seq)
    print(f"SASRec output shape: {output.shape}")