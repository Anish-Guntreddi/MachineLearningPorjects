# Recommender System Project - MovieLens / Amazon Reviews

## 1. Problem Definition & Use Case

**Problem:** Predict user preferences and recommend relevant items from vast catalogs while addressing the cold start problem, handling sparse data, and balancing exploration vs exploitation in dynamic environments.

**Use Case:** Recommender systems drive engagement and revenue across digital platforms:
- E-commerce product recommendations (Amazon, eBay)
- Content streaming platforms (Netflix, Spotify, YouTube)
- Social media feed curation (Facebook, Instagram, TikTok)
- News and article recommendations (Medium, Reddit)
- Job and professional networking (LinkedIn)
- Dating and social matching (Tinder, Bumble)
- Restaurant and local business discovery (Yelp, Uber Eats)
- Online learning course suggestions (Coursera, Udemy)

**Business Impact:** Effective recommendation systems increase user engagement by 35%, boost conversion rates by 20%, and account for 35% of Amazon's revenue and 80% of Netflix's viewing hours.

## 2. Dataset Acquisition & Preprocessing

### Primary Datasets
- **MovieLens**: Movie ratings from 100K to 27M ratings across multiple sizes
  ```python
  from surprise import Dataset
  data = Dataset.load_builtin('ml-100k')  # or ml-1m, ml-10m, ml-25m
  ```
- **Amazon Product Reviews**: Multi-category product ratings and reviews
  ```python
  import pandas as pd
  # Download from: https://nijianmo.github.io/amazon/index.html
  df = pd.read_json('Movies_and_TV_5.json.gz', lines=True)
  ```
- **Last.fm**: Music listening and artist preferences
  ```python
  import requests
  # API access for real-time data
  lastfm_data = requests.get('http://ws.audioscrobbler.com/2.0/')
  ```
- **Yelp Dataset**: Business reviews and ratings
  ```bash
  # Download from Yelp Dataset Challenge
  wget https://www.yelp.com/dataset/download
  ```

### Data Schema
```python
{
    'user_id': str,           # Unique user identifier
    'item_id': str,           # Unique item identifier  
    'rating': float,          # Explicit rating (1-5 scale)
    'timestamp': int,         # Unix timestamp
    'review_text': str,       # Optional review content
    'item_features': {
        'title': str,
        'category': List[str],
        'price': float,
        'brand': str,
        'description': str,
    },
    'user_features': {
        'age': int,
        'gender': str,
        'location': str,
        'occupation': str,
    },
    'implicit_feedback': {
        'views': int,
        'clicks': int,
        'time_spent': float,
        'purchases': int,
    }
}
```

### Preprocessing Pipeline
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.sparse import csr_matrix
import implicit

def load_and_preprocess_movielens(data_path='ml-100k/u.data'):
    """Load and preprocess MovieLens dataset"""
    # Load ratings data
    ratings = pd.read_csv(
        data_path, 
        sep='\t', 
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    
    # Load movie metadata
    movies = pd.read_csv(
        'ml-100k/u.item', 
        sep='|', 
        encoding='latin-1',
        names=['movie_id', 'title', 'release_date', 'video_date', 'url'] + 
              [f'genre_{i}' for i in range(19)]
    )
    
    # Load user demographics
    users = pd.read_csv(
        'ml-100k/u.user',
        sep='|',
        names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
    )
    
    return ratings, movies, users

def create_interaction_matrix(ratings_df, min_interactions=5):
    """Create user-item interaction matrix"""
    # Filter users and items with minimum interactions
    user_counts = ratings_df['user_id'].value_counts()
    item_counts = ratings_df['item_id'].value_counts()
    
    valid_users = user_counts[user_counts >= min_interactions].index
    valid_items = item_counts[item_counts >= min_interactions].index
    
    filtered_ratings = ratings_df[
        (ratings_df['user_id'].isin(valid_users)) &
        (ratings_df['item_id'].isin(valid_items))
    ].copy()
    
    # Create mappings
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    filtered_ratings['user_idx'] = user_encoder.fit_transform(filtered_ratings['user_id'])
    filtered_ratings['item_idx'] = item_encoder.fit_transform(filtered_ratings['item_id'])
    
    # Create sparse matrix
    n_users = len(user_encoder.classes_)
    n_items = len(item_encoder.classes_)
    
    interaction_matrix = csr_matrix(
        (filtered_ratings['rating'].values,
         (filtered_ratings['user_idx'].values, filtered_ratings['item_idx'].values)),
        shape=(n_users, n_items)
    )
    
    return interaction_matrix, user_encoder, item_encoder, filtered_ratings

def create_content_features(movies_df):
    """Extract content-based features from movies"""
    features = {}
    
    # Genre features (one-hot encoding)
    genre_cols = [col for col in movies_df.columns if col.startswith('genre_')]
    features['genres'] = movies_df[genre_cols].values
    
    # Release year extraction
    movies_df['release_year'] = pd.to_datetime(
        movies_df['release_date'], 
        format='%d-%b-%Y', 
        errors='coerce'
    ).dt.year
    
    # Normalize release year
    scaler = StandardScaler()
    features['release_year'] = scaler.fit_transform(
        movies_df[['release_year']].fillna(movies_df['release_year'].mean())
    )
    
    # Title text features (TF-IDF)
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    features['title_tfidf'] = tfidf.fit_transform(movies_df['title']).toarray()
    
    return features

def temporal_split(ratings_df, test_ratio=0.2):
    """Split data temporally for realistic evaluation"""
    # Sort by timestamp
    ratings_sorted = ratings_df.sort_values('timestamp')
    
    # Calculate split point
    split_idx = int(len(ratings_sorted) * (1 - test_ratio))
    
    train_data = ratings_sorted.iloc[:split_idx]
    test_data = ratings_sorted.iloc[split_idx:]
    
    return train_data, test_data

def create_negative_samples(train_data, neg_ratio=4):
    """Create negative samples for implicit feedback"""
    user_item_set = set(zip(train_data['user_idx'], train_data['item_idx']))
    
    negative_samples = []
    n_users = train_data['user_idx'].max() + 1
    n_items = train_data['item_idx'].max() + 1
    
    for user_id in range(n_users):
        # Get positive items for this user
        user_positives = set(train_data[train_data['user_idx'] == user_id]['item_idx'])
        n_positives = len(user_positives)
        
        # Sample negative items
        n_negatives = n_positives * neg_ratio
        all_items = set(range(n_items))
        candidate_negatives = all_items - user_positives
        
        if len(candidate_negatives) >= n_negatives:
            negatives = np.random.choice(
                list(candidate_negatives), 
                size=n_negatives, 
                replace=False
            )
            
            for item_id in negatives:
                negative_samples.append({
                    'user_idx': user_id,
                    'item_idx': item_id,
                    'rating': 0  # Implicit negative
                })
    
    return pd.DataFrame(negative_samples)
```

### Feature Engineering
- **User profiling**: Demographic features, interaction patterns, temporal behavior
- **Item features**: Content metadata, popularity metrics, category embeddings
- **Contextual features**: Time of day, seasonality, device type, location
- **Interaction features**: Rating variance, review sentiment, recency
- **Graph features**: User similarity, item similarity, clustering features

## 3. Baseline Models

### Memory-Based Collaborative Filtering
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class UserBasedCF:
    def __init__(self, k=50, similarity_metric='cosine'):
        self.k = k  # Number of similar users
        self.similarity_metric = similarity_metric
        self.user_similarity = None
        self.interaction_matrix = None
        
    def fit(self, interaction_matrix):
        """Fit user-based collaborative filtering"""
        self.interaction_matrix = interaction_matrix
        
        # Compute user similarity matrix
        if self.similarity_metric == 'cosine':
            self.user_similarity = cosine_similarity(interaction_matrix)
        elif self.similarity_metric == 'pearson':
            # Center the data first
            user_means = np.array(interaction_matrix.mean(axis=1)).flatten()
            centered_matrix = interaction_matrix.toarray() - user_means[:, np.newaxis]
            self.user_similarity = np.corrcoef(centered_matrix)
            
        # Set diagonal to 0 (user isn't similar to themselves for recommendation)
        np.fill_diagonal(self.user_similarity, 0)
        
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if self.user_similarity is None:
            raise ValueError("Model must be fitted first")
            
        # Get k most similar users who rated this item
        item_raters = np.nonzero(self.interaction_matrix[:, item_id])[0]
        
        if len(item_raters) == 0:
            # No one rated this item, return global mean
            return self.interaction_matrix.data.mean()
        
        # Get similarities to users who rated this item
        similarities = self.user_similarity[user_id, item_raters]
        
        # Get top-k similar users
        top_k_indices = np.argsort(similarities)[-self.k:]
        top_k_users = item_raters[top_k_indices]
        top_k_similarities = similarities[top_k_indices]
        
        # Remove users with zero or negative similarity
        positive_mask = top_k_similarities > 0
        if not np.any(positive_mask):
            return self.interaction_matrix.data.mean()
            
        top_k_users = top_k_users[positive_mask]
        top_k_similarities = top_k_similarities[positive_mask]
        
        # Get ratings from similar users
        ratings = self.interaction_matrix[top_k_users, item_id].toarray().flatten()
        
        # Weighted average prediction
        if np.sum(top_k_similarities) == 0:
            return self.interaction_matrix.data.mean()
            
        prediction = np.sum(ratings * top_k_similarities) / np.sum(top_k_similarities)
        return prediction
    
    def recommend(self, user_id, n_recommendations=10):
        """Recommend top-N items for user"""
        # Get items not yet rated by user
        user_ratings = self.interaction_matrix[user_id].toarray().flatten()
        unrated_items = np.where(user_ratings == 0)[0]
        
        # Predict ratings for unrated items
        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating and return top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

class ItemBasedCF:
    def __init__(self, k=50):
        self.k = k
        self.item_similarity = None
        self.interaction_matrix = None
        
    def fit(self, interaction_matrix):
        """Fit item-based collaborative filtering"""
        self.interaction_matrix = interaction_matrix
        
        # Compute item similarity matrix
        self.item_similarity = cosine_similarity(interaction_matrix.T)
        np.fill_diagonal(self.item_similarity, 0)
        
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        # Get items rated by this user
        user_ratings = self.interaction_matrix[user_id].toarray().flatten()
        rated_items = np.nonzero(user_ratings)[0]
        
        if len(rated_items) == 0:
            return self.interaction_matrix.data.mean()
        
        # Get similarities to rated items
        similarities = self.item_similarity[item_id, rated_items]
        
        # Get top-k similar items
        top_k_indices = np.argsort(similarities)[-self.k:]
        top_k_items = rated_items[top_k_indices]
        top_k_similarities = similarities[top_k_indices]
        
        # Remove items with zero or negative similarity
        positive_mask = top_k_similarities > 0
        if not np.any(positive_mask):
            return self.interaction_matrix.data.mean()
            
        top_k_items = top_k_items[positive_mask]
        top_k_similarities = top_k_similarities[positive_mask]
        
        # Get user's ratings for similar items
        ratings = user_ratings[top_k_items]
        
        # Weighted average prediction
        if np.sum(top_k_similarities) == 0:
            return self.interaction_matrix.data.mean()
            
        prediction = np.sum(ratings * top_k_similarities) / np.sum(top_k_similarities)
        return prediction
```
**Expected Performance:** RMSE 0.9-1.1 on MovieLens, good interpretability

### Matrix Factorization (SVD)
```python
import numpy as np
from sklearn.decomposition import TruncatedSVD

class MatrixFactorization:
    def __init__(self, n_factors=50, learning_rate=0.01, reg_lambda=0.01, n_epochs=100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.n_epochs = n_epochs
        
    def fit(self, interaction_matrix):
        """Fit matrix factorization using SGD"""
        self.n_users, self.n_items = interaction_matrix.shape
        
        # Initialize latent factors
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        
        # Initialize biases
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        self.global_bias = interaction_matrix.data.mean()
        
        # Convert sparse matrix to list of (user, item, rating) tuples
        interactions = []
        for user_id in range(self.n_users):
            for item_id in range(self.n_items):
                if interaction_matrix[user_id, item_id] != 0:
                    interactions.append((user_id, item_id, interaction_matrix[user_id, item_id]))
        
        # SGD training
        for epoch in range(self.n_epochs):
            np.random.shuffle(interactions)
            
            for user_id, item_id, rating in interactions:
                # Predict rating
                pred = self.predict(user_id, item_id)
                error = rating - pred
                
                # Update factors and biases
                user_factor = self.user_factors[user_id].copy()
                item_factor = self.item_factors[item_id].copy()
                
                # Update user and item factors
                self.user_factors[user_id] += self.learning_rate * (
                    error * item_factor - self.reg_lambda * user_factor
                )
                self.item_factors[item_id] += self.learning_rate * (
                    error * user_factor - self.reg_lambda * item_factor
                )
                
                # Update biases
                self.user_biases[user_id] += self.learning_rate * (
                    error - self.reg_lambda * self.user_biases[user_id]
                )
                self.item_biases[item_id] += self.learning_rate * (
                    error - self.reg_lambda * self.item_biases[item_id]
                )
            
            # Decay learning rate
            self.learning_rate *= 0.99
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        pred = (self.global_bias + 
                self.user_biases[user_id] + 
                self.item_biases[item_id] +
                np.dot(self.user_factors[user_id], self.item_factors[item_id]))
        return pred
    
    def recommend(self, user_id, n_recommendations=10):
        """Recommend top-N items for user"""
        # Predict ratings for all items
        predictions = []
        for item_id in range(self.n_items):
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating and return top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

# Alternative: Using Surprise library for easier implementation
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

def train_surprise_svd(ratings_df):
    """Train SVD using Surprise library"""
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
    
    trainset, testset = train_test_split(data, test_size=0.2)
    
    # Train SVD
    svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    svd.fit(trainset)
    
    return svd, testset
```
**Expected Performance:** RMSE 0.85-0.95 on MovieLens, scalable to large datasets

## 4. Advanced/Stretch Models

### Deep Learning Approaches

1. **Neural Collaborative Filtering (NCF)**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, hidden_dims=[128, 64]):
        super().__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # MLP layers
        mlp_input_dim = embedding_dim * 2
        self.mlp_layers = nn.ModuleList()
        
        prev_dim = mlp_input_dim
        for hidden_dim in hidden_dims:
            self.mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
    def forward(self, user_ids, item_ids):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=1)
        
        # Pass through MLP
        for layer in self.mlp_layers:
            x = layer(x)
        
        # Output prediction
        rating = self.output_layer(x)
        return rating.squeeze()

class NCFTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            user_ids, item_ids, ratings = batch
            
            # Forward pass
            predictions = self.model(user_ids, item_ids)
            loss = self.criterion(predictions, ratings.float())
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                user_ids, item_ids, ratings = batch
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings.float())
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
```

2. **Variational Autoencoders for Collaborative Filtering (VAE-CF)**
```python
class VariationalAutoEncoder(nn.Module):
    def __init__(self, n_items, hidden_dim=600, latent_dim=200):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_items, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_items)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss with KL divergence regularization"""
    # Reconstruction loss (multinomial likelihood)
    recon_loss = -torch.sum(F.log_softmax(recon_x, dim=1) * x, dim=1).mean()
    
    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    return recon_loss + beta * kld_loss
```

3. **Graph Neural Networks for Recommendations**
```python
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool

class GraphRecommender(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Node embeddings (users + items)
        total_nodes = n_users + n_items
        self.node_embedding = nn.Embedding(total_nodes, embedding_dim)
        
        # GCN layers
        self.conv_layers = nn.ModuleList([
            GCNConv(embedding_dim, embedding_dim) for _ in range(n_layers)
        ])
        
        # Prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        
    def forward(self, edge_index, user_ids, item_ids):
        # Get all node embeddings
        x = self.node_embedding.weight
        
        # Apply GCN layers
        for conv in self.conv_layers:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, training=self.training)
        
        # Get user and item embeddings
        user_emb = x[user_ids]
        item_emb = x[self.n_users + item_ids]  # Items start after users
        
        # Concatenate and predict
        combined = torch.cat([user_emb, item_emb], dim=1)
        rating = self.predictor(combined)
        
        return rating.squeeze()

def create_bipartite_graph(interactions_df, n_users, n_items):
    """Create bipartite graph from user-item interactions"""
    # Create edge index for PyTorch Geometric
    user_nodes = interactions_df['user_idx'].values
    item_nodes = interactions_df['item_idx'].values + n_users  # Offset items
    
    # Create bidirectional edges
    edge_index = torch.tensor([
        np.concatenate([user_nodes, item_nodes]),
        np.concatenate([item_nodes, user_nodes])
    ], dtype=torch.long)
    
    return edge_index
```

4. **Multi-Armed Bandit for Exploration-Exploitation**
```python
class ContextualBandit:
    def __init__(self, n_arms, context_dim, alpha=1.0):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        
        # Initialize parameters
        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        
    def select_arm(self, context):
        """Select arm using Upper Confidence Bound"""
        p = np.zeros(self.n_arms)
        
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            
            # Confidence interval
            confidence = self.alpha * np.sqrt(context.T @ A_inv @ context)
            p[a] = theta.T @ context + confidence
        
        return np.argmax(p)
    
    def update(self, context, arm, reward):
        """Update parameters based on observed reward"""
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

class ThompsonSamplingRecommender:
    def __init__(self, n_items, feature_dim):
        self.n_items = n_items
        self.feature_dim = feature_dim
        
        # Prior parameters
        self.alpha = np.ones(n_items)  # Success count
        self.beta = np.ones(n_items)   # Failure count
        
    def recommend(self, n_recommendations=10):
        """Recommend items using Thompson Sampling"""
        # Sample from Beta distributions
        sampled_rewards = np.random.beta(self.alpha, self.beta)
        
        # Return top-k items based on sampled rewards
        top_items = np.argsort(sampled_rewards)[-n_recommendations:][::-1]
        return top_items
    
    def update_reward(self, item_id, reward):
        """Update posterior based on user feedback"""
        if reward > 0:
            self.alpha[item_id] += 1
        else:
            self.beta[item_id] += 1
```

**Target Performance:** RMSE < 0.8 on MovieLens, handling millions of users/items

## 5. Training Details

### Data Loading and Batching
```python
import torch
from torch.utils.data import Dataset, DataLoader

class RecommenderDataset(Dataset):
    def __init__(self, interactions_df, negative_sampling=True, neg_ratio=4):
        self.interactions = interactions_df
        self.negative_sampling = negative_sampling
        self.neg_ratio = neg_ratio
        
        if negative_sampling:
            self.create_negative_samples()
        
    def create_negative_samples(self):
        """Create negative samples for implicit feedback"""
        # Get all user-item pairs
        all_interactions = set(
            zip(self.interactions['user_idx'], self.interactions['item_idx'])
        )
        
        negative_samples = []
        n_users = self.interactions['user_idx'].max() + 1
        n_items = self.interactions['item_idx'].max() + 1
        
        for user_id in range(n_users):
            user_items = set(
                self.interactions[self.interactions['user_idx'] == user_id]['item_idx']
            )
            n_positives = len(user_items)
            
            # Sample negative items
            all_items = set(range(n_items))
            candidate_negatives = all_items - user_items
            
            if len(candidate_negatives) >= n_positives * self.neg_ratio:
                negatives = np.random.choice(
                    list(candidate_negatives),
                    size=n_positives * self.neg_ratio,
                    replace=False
                )
                
                for item_id in negatives:
                    negative_samples.append({
                        'user_idx': user_id,
                        'item_idx': item_id,
                        'rating': 0
                    })
        
        # Combine positive and negative samples
        positive_samples = self.interactions.copy()
        positive_samples['rating'] = 1  # Convert to implicit feedback
        
        all_samples = pd.concat([
            positive_samples[['user_idx', 'item_idx', 'rating']],
            pd.DataFrame(negative_samples)
        ]).reset_index(drop=True)
        
        self.interactions = all_samples
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        return (
            torch.tensor(row['user_idx'], dtype=torch.long),
            torch.tensor(row['item_idx'], dtype=torch.long),
            torch.tensor(row['rating'], dtype=torch.float)
        )

def create_dataloaders(train_df, val_df, batch_size=1024):
    """Create train and validation dataloaders"""
    train_dataset = RecommenderDataset(train_df, negative_sampling=True)
    val_dataset = RecommenderDataset(val_df, negative_sampling=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader
```

### Training Configuration
```python
training_config = {
    'model_type': 'neural_collaborative_filtering',
    'embedding_dim': 64,
    'hidden_dims': [128, 64, 32],
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'batch_size': 1024,
    'num_epochs': 100,
    'early_stopping_patience': 10,
    'negative_sampling_ratio': 4,
    'dropout_rate': 0.2,
    'scheduler': 'cosine_annealing',
    'gradient_clip_norm': 1.0,
    'loss_function': 'bce',  # binary cross-entropy for implicit feedback
}

class RecommenderTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Loss function
        if config['loss_function'] == 'mse':
            self.criterion = nn.MSELoss()
        elif config['loss_function'] == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        if config['scheduler'] == 'cosine_annealing':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config['num_epochs']
            )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            user_ids, item_ids, ratings = batch
            
            # Forward pass
            predictions = self.model(user_ids, item_ids)
            loss = self.criterion(predictions, ratings)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['gradient_clip_norm']
            )
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                user_ids, item_ids, ratings = batch
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch}")
                break
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
```

### Advanced Training Techniques
```python
# Learning rate scheduling with warm restarts
class CosineAnnealingWarmRestarts:
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        
    def step(self):
        if self.T_cur == self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult
        
        eta_t = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + np.cos(np.pi * self.T_cur / self.T_i)
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = eta_t
        
        self.T_cur += 1

# Regularization techniques
class RegularizedRecommender(nn.Module):
    def __init__(self, base_model, l1_lambda=0.0, l2_lambda=0.01):
        super().__init__()
        self.base_model = base_model
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def regularization_loss(self):
        """Compute regularization loss"""
        l1_loss = 0
        l2_loss = 0
        
        for param in self.base_model.parameters():
            l1_loss += torch.sum(torch.abs(param))
            l2_loss += torch.sum(param ** 2)
        
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss

# Curriculum learning for recommender systems
def curriculum_learning_scheduler(epoch, max_epochs):
    """Gradually increase difficulty of negative samples"""
    # Start with easy negatives (popular items)
    # Progress to hard negatives (similar items)
    difficulty_ratio = epoch / max_epochs
    return difficulty_ratio
```

## 6. Evaluation Metrics & Validation Strategy

### Core Metrics
```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

def rmse(predictions, targets):
    """Root Mean Square Error"""
    return np.sqrt(mean_squared_error(targets, predictions))

def mae(predictions, targets):
    """Mean Absolute Error"""
    return mean_absolute_error(targets, predictions)

def precision_at_k(recommended_items, relevant_items, k):
    """Precision@K"""
    recommended_k = recommended_items[:k]
    relevant_recommended = set(recommended_k) & set(relevant_items)
    return len(relevant_recommended) / k if k > 0 else 0

def recall_at_k(recommended_items, relevant_items, k):
    """Recall@K"""
    recommended_k = recommended_items[:k]
    relevant_recommended = set(recommended_k) & set(relevant_items)
    return len(relevant_recommended) / len(relevant_items) if len(relevant_items) > 0 else 0

def ndcg_at_k(recommended_items, relevant_items, k):
    """Normalized Discounted Cumulative Gain@K"""
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.
    
    # Create relevance scores for recommended items
    relevance_scores = [1 if item in relevant_items else 0 for item in recommended_items[:k]]
    
    # Calculate DCG
    dcg = dcg_at_k(relevance_scores, k)
    
    # Calculate IDCG (ideal DCG)
    ideal_relevance = [1] * min(len(relevant_items), k)
    idcg = dcg_at_k(ideal_relevance, k)
    
    return dcg / idcg if idcg > 0 else 0

def mean_reciprocal_rank(recommended_items_list, relevant_items_list):
    """Mean Reciprocal Rank"""
    reciprocal_ranks = []
    
    for recommended_items, relevant_items in zip(recommended_items_list, relevant_items_list):
        relevant_set = set(relevant_items)
        
        for i, item in enumerate(recommended_items):
            if item in relevant_set:
                reciprocal_ranks.append(1.0 / (i + 1))
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)

def hit_rate_at_k(recommended_items_list, relevant_items_list, k):
    """Hit Rate@K (fraction of users with at least one relevant item in top-k)"""
    hits = 0
    
    for recommended_items, relevant_items in zip(recommended_items_list, relevant_items_list):
        recommended_k = set(recommended_items[:k])
        relevant_set = set(relevant_items)
        
        if recommended_k & relevant_set:
            hits += 1
    
    return hits / len(recommended_items_list)

def coverage(recommended_items_list, total_items):
    """Catalog coverage (fraction of items that appear in recommendations)"""
    all_recommended = set()
    for recommended_items in recommended_items_list:
        all_recommended.update(recommended_items)
    
    return len(all_recommended) / total_items

def diversity_at_k(recommended_items_list, item_features, k):
    """Average pairwise diversity in recommendations"""
    diversities = []
    
    for recommended_items in recommended_items_list:
        recommended_k = recommended_items[:k]
        
        if len(recommended_k) < 2:
            continue
        
        pairwise_distances = []
        for i in range(len(recommended_k)):
            for j in range(i + 1, len(recommended_k)):
                item1, item2 = recommended_k[i], recommended_k[j]
                
                # Calculate distance between items (cosine similarity of features)
                if item1 in item_features and item2 in item_features:
                    feat1 = item_features[item1]
                    feat2 = item_features[item2]
                    
                    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
                    distance = 1 - similarity
                    pairwise_distances.append(distance)
        
        if pairwise_distances:
            diversities.append(np.mean(pairwise_distances))
    
    return np.mean(diversities) if diversities else 0

class RecommenderEvaluator:
    def __init__(self, model, test_interactions, item_features=None):
        self.model = model
        self.test_interactions = test_interactions
        self.item_features = item_features
        
    def evaluate_user(self, user_id, k_values=[5, 10, 20]):
        """Evaluate recommendations for a single user"""
        # Get test items for this user
        user_test_items = self.test_interactions[
            self.test_interactions['user_idx'] == user_id
        ]['item_idx'].tolist()
        
        if not user_test_items:
            return None
        
        # Get recommendations
        recommendations = self.model.recommend(user_id, n_recommendations=max(k_values))
        recommended_items = [item_id for item_id, _ in recommendations]
        
        # Calculate metrics for each k
        results = {}
        for k in k_values:
            results[f'precision@{k}'] = precision_at_k(recommended_items, user_test_items, k)
            results[f'recall@{k}'] = recall_at_k(recommended_items, user_test_items, k)
            results[f'ndcg@{k}'] = ndcg_at_k(recommended_items, user_test_items, k)
        
        return results
    
    def evaluate_all_users(self, k_values=[5, 10, 20]):
        """Evaluate recommendations for all test users"""
        all_results = []
        all_recommendations = []
        all_relevant_items = []
        
        unique_users = self.test_interactions['user_idx'].unique()
        
        for user_id in unique_users:
            user_results = self.evaluate_user(user_id, k_values)
            if user_results:
                all_results.append(user_results)
                
                # Collect for aggregate metrics
                recommendations = self.model.recommend(user_id, n_recommendations=max(k_values))
                recommended_items = [item_id for item_id, _ in recommendations]
                
                user_test_items = self.test_interactions[
                    self.test_interactions['user_idx'] == user_id
                ]['item_idx'].tolist()
                
                all_recommendations.append(recommended_items)
                all_relevant_items.append(user_test_items)
        
        # Aggregate results
        aggregated_results = {}
        for metric in all_results[0].keys():
            values = [result[metric] for result in all_results]
            aggregated_results[metric] = np.mean(values)
        
        # Add aggregate metrics
        for k in k_values:
            aggregated_results[f'hit_rate@{k}'] = hit_rate_at_k(
                all_recommendations, all_relevant_items, k
            )
        
        aggregated_results['mrr'] = mean_reciprocal_rank(all_recommendations, all_relevant_items)
        aggregated_results['coverage'] = coverage(all_recommendations, self.model.n_items)
        
        if self.item_features:
            for k in k_values:
                aggregated_results[f'diversity@{k}'] = diversity_at_k(
                    all_recommendations, self.item_features, k
                )
        
        return aggregated_results
```

### Validation Strategy
- **Temporal splitting**: Train on historical data, test on future interactions
- **User-based splitting**: Hold out some users entirely for testing
- **Item-based splitting**: Test on new items (cold start evaluation)
- **Leave-one-out**: For each user, predict the last interaction
- **Cross-validation**: K-fold with user/time-aware splitting

### Advanced Evaluation
```python
# Cold start evaluation
def cold_start_evaluation(model, new_users_data, new_items_data):
    """Evaluate performance on new users and items"""
    results = {}
    
    # New users (user cold start)
    if new_users_data is not None:
        new_user_metrics = []
        for user_id in new_users_data['user_idx'].unique():
            # Recommend items for new user
            recommendations = model.recommend_new_user(
                user_profile=new_users_data[new_users_data['user_idx'] == user_id]
            )
            # Evaluate against test interactions
            # ... evaluation logic ...
        
        results['new_user_performance'] = np.mean(new_user_metrics)
    
    # New items (item cold start)
    if new_items_data is not None:
        # Evaluate how well new items get recommended
        new_item_coverage = evaluate_new_item_coverage(model, new_items_data)
        results['new_item_coverage'] = new_item_coverage
    
    return results

# A/B testing framework
class ABTestEvaluator:
    def __init__(self, control_model, treatment_model):
        self.control_model = control_model
        self.treatment_model = treatment_model
        
    def run_ab_test(self, test_users, test_duration_days=7):
        """Run A/B test comparing two recommender models"""
        # Randomly assign users to control/treatment
        np.random.shuffle(test_users)
        split_point = len(test_users) // 2
        
        control_users = test_users[:split_point]
        treatment_users = test_users[split_point:]
        
        # Collect metrics for both groups
        control_metrics = self.collect_metrics(self.control_model, control_users)
        treatment_metrics = self.collect_metrics(self.treatment_model, treatment_users)
        
        # Statistical significance testing
        from scipy import stats
        
        results = {}
        for metric in control_metrics.keys():
            control_values = control_metrics[metric]
            treatment_values = treatment_metrics[metric]
            
            # T-test
            t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
            
            results[metric] = {
                'control_mean': np.mean(control_values),
                'treatment_mean': np.mean(treatment_values),
                'lift': (np.mean(treatment_values) - np.mean(control_values)) / np.mean(control_values),
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return results

# Fairness evaluation
def evaluate_fairness(model, test_data, protected_attribute='gender'):
    """Evaluate fairness across demographic groups"""
    fairness_metrics = {}
    
    unique_groups = test_data[protected_attribute].unique()
    
    for group in unique_groups:
        group_data = test_data[test_data[protected_attribute] == group]
        group_users = group_data['user_idx'].unique()
        
        # Evaluate performance for this group
        group_metrics = []
        for user_id in group_users:
            user_metrics = evaluate_user_recommendations(model, user_id, group_data)
            if user_metrics:
                group_metrics.append(user_metrics)
        
        # Average metrics for this group
        if group_metrics:
            fairness_metrics[group] = {
                metric: np.mean([m[metric] for m in group_metrics])
                for metric in group_metrics[0].keys()
            }
    
    # Calculate fairness measures
    fairness_summary = {}
    for metric in fairness_metrics[unique_groups[0]].keys():
        group_values = [fairness_metrics[group][metric] for group in unique_groups]
        
        # Statistical parity difference
        fairness_summary[f'{metric}_parity_diff'] = max(group_values) - min(group_values)
        
        # Demographic parity ratio
        fairness_summary[f'{metric}_parity_ratio'] = min(group_values) / max(group_values)
    
    return fairness_metrics, fairness_summary
```

## 7. Experiment Tracking & Reproducibility

### Weights & Biases Integration
```python
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize experiment tracking
wandb.init(
    project="recommender-systems",
    config=training_config,
    tags=["neural-cf", "movielens", "collaborative-filtering"]
)

class RecommenderLoggingCallback:
    def __init__(self, model, val_loader, test_interactions, k_values=[5, 10, 20]):
        self.model = model
        self.val_loader = val_loader
        self.test_interactions = test_interactions
        self.k_values = k_values
        
    def on_epoch_end(self, epoch, train_loss, val_loss):
        """Log metrics at the end of each epoch"""
        metrics = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
        
        # Evaluate on test set every 10 epochs
        if epoch % 10 == 0:
            evaluator = RecommenderEvaluator(self.model, self.test_interactions)
            test_metrics = evaluator.evaluate_all_users(self.k_values)
            metrics.update(test_metrics)
            
            # Log sample recommendations
            self.log_sample_recommendations(epoch)
        
        wandb.log(metrics)
    
    def log_sample_recommendations(self, epoch, n_users=5):
        """Log sample recommendations for analysis"""
        sample_users = np.random.choice(
            self.test_interactions['user_idx'].unique(), 
            size=min(n_users, len(self.test_interactions['user_idx'].unique())),
            replace=False
        )
        
        recommendations_table = []
        
        for user_id in sample_users:
            # Get recommendations
            recommendations = self.model.recommend(user_id, n_recommendations=10)
            recommended_items = [item_id for item_id, score in recommendations]
            recommended_scores = [score for item_id, score in recommendations]
            
            # Get actual test items
            actual_items = self.test_interactions[
                self.test_interactions['user_idx'] == user_id
            ]['item_idx'].tolist()
            
            # Calculate metrics
            precision_5 = precision_at_k(recommended_items, actual_items, 5)
            ndcg_10 = ndcg_at_k(recommended_items, actual_items, 10)
            
            recommendations_table.append([
                user_id,
                recommended_items[:5],
                recommended_scores[:5],
                actual_items,
                precision_5,
                ndcg_10
            ])
        
        # Log as table
        wandb.log({
            f"recommendations_epoch_{epoch}": wandb.Table(
                columns=[
                    "user_id", "recommended_items", "scores", 
                    "actual_items", "precision@5", "ndcg@10"
                ],
                data=recommendations_table
            )
        })
    
    def log_embedding_visualization(self, epoch):
        """Log embedding visualizations"""
        if hasattr(self.model, 'user_embedding'):
            # Get user embeddings
            user_embeddings = self.model.user_embedding.weight.data.cpu().numpy()
            
            # Use t-SNE for dimensionality reduction
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(user_embeddings[:1000])  # Sample for speed
            
            # Create scatter plot
            plt.figure(figsize=(10, 8))
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
            plt.title(f'User Embeddings (t-SNE) - Epoch {epoch}')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            
            # Log plot
            wandb.log({f"user_embeddings_epoch_{epoch}": wandb.Image(plt)})
            plt.close()

# Enhanced trainer with logging
class LoggingRecommenderTrainer(RecommenderTrainer):
    def __init__(self, model, config, logging_callback=None):
        super().__init__(model, config)
        self.logging_callback = logging_callback
        
    def train(self, train_loader, val_loader):
        """Training loop with enhanced logging"""
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Log metrics
            if self.logging_callback:
                self.logging_callback.on_epoch_end(epoch, train_loss, val_loss)
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
                wandb.log({'learning_rate': self.scheduler.get_last_lr()[0]})
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pt')
                wandb.save('best_model.pt')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch}")
                break
```

### MLflow Integration
```python
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

mlflow.set_experiment("movielens-recommender")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "model_type": training_config['model_type'],
        "embedding_dim": training_config['embedding_dim'],
        "learning_rate": training_config['learning_rate'],
        "batch_size": training_config['batch_size'],
        "negative_sampling_ratio": training_config['negative_sampling_ratio'],
        "dataset": "movielens-100k"
    })
    
    # Train model
    trainer = LoggingRecommenderTrainer(model, training_config)
    trainer.train(train_loader, val_loader)
    
    # Final evaluation
    evaluator = RecommenderEvaluator(model, test_interactions)
    final_metrics = evaluator.evaluate_all_users()
    
    # Log metrics
    for metric_name, value in final_metrics.items():
        mlflow.log_metric(metric_name, value)
    
    # Log model
    mlflow.pytorch.log_model(
        model,
        "recommender-model",
        registered_model_name="MovieLensRecommender"
    )
    
    # Log evaluation artifacts
    mlflow.log_artifacts("evaluation_results/", "evaluation")
```

### Experiment Configuration
```yaml
# experiment_config.yaml
experiment:
  name: "neural-collaborative-filtering"
  description: "NCF model on MovieLens-100K dataset"
  tags: ["recommender", "collaborative-filtering", "neural-networks"]

model:
  type: "neural_collaborative_filtering"
  embedding_dim: 64
  hidden_dims: [128, 64, 32]
  dropout_rate: 0.2
  activation: "relu"

data:
  dataset: "movielens-100k"
  min_interactions: 5
  test_ratio: 0.2
  negative_sampling: true
  negative_ratio: 4
  
training:
  batch_size: 1024
  learning_rate: 0.001
  weight_decay: 1e-5
  num_epochs: 100
  early_stopping_patience: 10
  gradient_clip_norm: 1.0
  scheduler:
    type: "cosine_annealing"
    T_max: 100

evaluation:
  k_values: [5, 10, 20]
  metrics: ["precision", "recall", "ndcg", "hit_rate", "mrr"]
  cold_start_evaluation: true
  fairness_evaluation: true
  
reproducibility:
  random_seed: 42
  cuda_deterministic: true
  workers: 4
```

## 8. Deployment Pathway

### Option 1: Real-time Recommendation API
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import redis
import pickle
from typing import List, Optional

app = FastAPI(title="Recommendation API")

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('best_model.pt', map_location=device)
model.eval()

# Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

class User(BaseModel):
    user_id: str
    features: Optional[dict] = None

class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = 10
    exclude_items: Optional[List[str]] = None
    context: Optional[dict] = None

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[dict]
    explanation: Optional[str] = None
    timestamp: float

class RecommendationService:
    def __init__(self, model, item_metadata=None):
        self.model = model
        self.item_metadata = item_metadata or {}
        
    def get_user_recommendations(self, user_id: str, num_recommendations: int = 10, 
                               exclude_items: List[str] = None) -> List[dict]:
        """Generate recommendations for a user"""
        
        # Check cache first
        cache_key = f"recommendations:{user_id}:{num_recommendations}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            return pickle.loads(cached_result)
        
        # Convert user_id to internal format
        user_idx = self.user_encoder.transform([user_id])[0] if hasattr(self, 'user_encoder') else int(user_id)
        
        # Get all item predictions
        all_items = torch.arange(self.model.n_items)
        user_tensor = torch.tensor([user_idx] * len(all_items), dtype=torch.long)
        
        with torch.no_grad():
            predictions = self.model(user_tensor, all_items)
            scores = torch.sigmoid(predictions).cpu().numpy()
        
        # Create item-score pairs
        item_scores = list(zip(all_items.numpy(), scores))
        
        # Filter out excluded items
        if exclude_items:
            exclude_set = set([self.item_encoder.transform([item])[0] for item in exclude_items])
            item_scores = [(item, score) for item, score in item_scores if item not in exclude_set]
        
        # Sort by score and take top N
        item_scores.sort(key=lambda x: x[1], reverse=True)
        top_items = item_scores[:num_recommendations]
        
        # Format recommendations with metadata
        recommendations = []
        for item_idx, score in top_items:
            item_id = self.item_encoder.inverse_transform([item_idx])[0] if hasattr(self, 'item_encoder') else str(item_idx)
            
            rec = {
                'item_id': item_id,
                'score': float(score),
                'metadata': self.item_metadata.get(item_id, {})
            }
            recommendations.append(rec)
        
        # Cache results for 1 hour
        redis_client.setex(cache_key, 3600, pickle.dumps(recommendations))
        
        return recommendations
    
    def get_similar_items(self, item_id: str, num_similar: int = 10) -> List[dict]:
        """Find similar items using item embeddings"""
        if not hasattr(self.model, 'item_embedding'):
            raise ValueError("Model doesn't have item embeddings")
        
        item_idx = self.item_encoder.transform([item_id])[0]
        
        # Get item embedding
        item_embedding = self.model.item_embedding(torch.tensor([item_idx]))
        
        # Compute similarities with all items
        all_embeddings = self.model.item_embedding.weight
        similarities = torch.cosine_similarity(item_embedding, all_embeddings)
        
        # Get top similar items (excluding self)
        similarities[item_idx] = -1  # Exclude self
        top_indices = torch.topk(similarities, num_similar).indices
        
        similar_items = []
        for idx in top_indices:
            similar_item_id = self.item_encoder.inverse_transform([idx.item()])[0]
            similar_items.append({
                'item_id': similar_item_id,
                'similarity': float(similarities[idx]),
                'metadata': self.item_metadata.get(similar_item_id, {})
            })
        
        return similar_items

# Initialize service
rec_service = RecommendationService(model)

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations for user"""
    try:
        recommendations = rec_service.get_user_recommendations(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations,
            exclude_items=request.exclude_items or []
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            timestamp=time.time()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/similar/{item_id}")
async def get_similar_items(item_id: str, num_similar: int = 10):
    """Get items similar to the given item"""
    try:
        similar_items = rec_service.get_similar_items(item_id, num_similar)
        return {"item_id": item_id, "similar_items": similar_items}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def record_feedback(user_id: str, item_id: str, feedback: float):
    """Record user feedback for online learning"""
    try:
        # Store feedback in database
        feedback_data = {
            'user_id': user_id,
            'item_id': item_id,
            'feedback': feedback,
            'timestamp': time.time()
        }
        
        # Invalidate cache for this user
        cache_pattern = f"recommendations:{user_id}:*"
        for key in redis_client.scan_iter(match=cache_pattern):
            redis_client.delete(key)
        
        return {"status": "feedback recorded"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}
```

### Option 2: Batch Recommendation Pipeline
```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import pandas as pd

class BatchRecommendationPipeline:
    def __init__(self, model_path, output_path):
        self.model_path = model_path
        self.output_path = output_path
        
    def run_pipeline(self, user_data_path, item_data_path):
        """Run batch recommendation pipeline using Apache Beam"""
        
        pipeline_options = PipelineOptions([
            '--runner=DataflowRunner',
            '--project=your-project-id',
            '--region=us-central1',
            '--temp_location=gs://your-bucket/temp'
        ])
        
        with beam.Pipeline(options=pipeline_options) as pipeline:
            # Read user data
            users = (pipeline 
                    | 'Read Users' >> beam.io.ReadFromText(user_data_path)
                    | 'Parse Users' >> beam.Map(self.parse_user_data))
            
            # Read item data
            items = (pipeline 
                    | 'Read Items' >> beam.io.ReadFromText(item_data_path)
                    | 'Parse Items' >> beam.Map(self.parse_item_data))
            
            # Generate recommendations
            recommendations = (users 
                             | 'Generate Recommendations' >> beam.ParDo(
                                 GenerateRecommendationsDoFn(self.model_path)
                             ))
            
            # Write recommendations
            (recommendations 
             | 'Format Output' >> beam.Map(self.format_output)
             | 'Write Recommendations' >> beam.io.WriteToText(self.output_path))
    
    def parse_user_data(self, line):
        """Parse user data from input"""
        data = json.loads(line)
        return data['user_id'], data
    
    def parse_item_data(self, line):
        """Parse item data from input"""
        data = json.loads(line)
        return data['item_id'], data
    
    def format_output(self, recommendation):
        """Format recommendation for output"""
        return json.dumps(recommendation)

class GenerateRecommendationsDoFn(beam.DoFn):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
    
    def setup(self):
        """Load model once per worker"""
        self.model = torch.load(self.model_path, map_location='cpu')
        self.model.eval()
    
    def process(self, element):
        """Generate recommendations for a user"""
        user_id, user_data = element
        
        # Generate recommendations using the model
        recommendations = self.model.recommend(user_id, n_recommendations=50)
        
        yield {
            'user_id': user_id,
            'recommendations': recommendations,
            'timestamp': time.time()
        }

# Airflow DAG for daily batch recommendations
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

def run_daily_recommendations():
    """Daily batch recommendation job"""
    pipeline = BatchRecommendationPipeline(
        model_path='gs://your-bucket/models/latest_model.pt',
        output_path='gs://your-bucket/recommendations/{{ ds }}'
    )
    
    pipeline.run_pipeline(
        user_data_path='gs://your-bucket/users/{{ ds }}',
        item_data_path='gs://your-bucket/items/{{ ds }}'
    )

# Define DAG
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'daily_recommendations',
    default_args=default_args,
    description='Daily batch recommendation generation',
    schedule_interval='0 2 * * *',  # Run at 2 AM daily
    catchup=False
)

# Task to generate recommendations
generate_recs_task = PythonOperator(
    task_id='generate_recommendations',
    python_callable=run_daily_recommendations,
    dag=dag
)
```

### Option 3: Real-time Streaming Recommendations
```python
from kafka import KafkaConsumer, KafkaProducer
import json
import asyncio

class StreamingRecommendationService:
    def __init__(self, model, kafka_config):
        self.model = model
        self.kafka_config = kafka_config
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
    async def process_user_events(self):
        """Process real-time user events and generate recommendations"""
        consumer = KafkaConsumer(
            'user_events',
            bootstrap_servers=self.kafka_config['bootstrap_servers'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        for message in consumer:
            event = message.value
            
            if event['event_type'] == 'item_view':
                await self.handle_item_view(event)
            elif event['event_type'] == 'item_purchase':
                await self.handle_item_purchase(event)
            elif event['event_type'] == 'user_login':
                await self.handle_user_login(event)
    
    async def handle_item_view(self, event):
        """Handle item view event"""
        user_id = event['user_id']
        item_id = event['item_id']
        
        # Update user session context
        session_context = self.get_session_context(user_id)
        session_context['recent_views'].append(item_id)
        
        # Generate contextual recommendations
        recommendations = await self.generate_contextual_recommendations(
            user_id, session_context
        )
        
        # Send recommendations
        self.producer.send('recommendations', {
            'user_id': user_id,
            'recommendations': recommendations,
            'context': 'item_view',
            'timestamp': time.time()
        })
    
    async def generate_contextual_recommendations(self, user_id, context):
        """Generate recommendations based on current context"""
        # Use context to bias recommendations
        recent_items = context.get('recent_views', [])
        
        # Get base recommendations
        base_recs = self.model.recommend(user_id, n_recommendations=20)
        
        # Apply contextual filtering/reranking
        contextual_recs = self.apply_contextual_reranking(base_recs, context)
        
        return contextual_recs[:10]
    
    def apply_contextual_reranking(self, recommendations, context):
        """Rerank recommendations based on context"""
        # Simple example: boost items similar to recently viewed
        recent_items = context.get('recent_views', [])
        
        if not recent_items:
            return recommendations
        
        # Boost scores for items similar to recent views
        for rec in recommendations:
            similarity_boost = self.calculate_item_similarity(
                rec['item_id'], recent_items
            )
            rec['score'] *= (1 + similarity_boost)
        
        # Resort by updated scores
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations

# Example usage
async def main():
    kafka_config = {
        'bootstrap_servers': ['localhost:9092']
    }
    
    streaming_service = StreamingRecommendationService(model, kafka_config)
    await streaming_service.process_user_events()

if __name__ == "__main__":
    asyncio.run(main())
```

### Cloud Deployment Options
- **AWS SageMaker**: Real-time inference endpoints with auto-scaling
- **Google Cloud AI Platform**: Managed ML model serving
- **Azure ML**: Enterprise deployment with monitoring
- **Kubernetes**: Container orchestration for microservices architecture
- **Serverless**: AWS Lambda/Cloud Functions for event-driven recommendations

## 9. Extensions & Research Directions

### Advanced Techniques
1. **Multi-Task Learning**
   ```python
   class MultiTaskRecommender(nn.Module):
       def __init__(self, n_users, n_items, embedding_dim=64):
           super().__init__()
           
           # Shared embeddings
           self.user_embedding = nn.Embedding(n_users, embedding_dim)
           self.item_embedding = nn.Embedding(n_items, embedding_dim)
           
           # Task-specific heads
           self.rating_predictor = nn.Linear(embedding_dim * 2, 1)
           self.click_predictor = nn.Linear(embedding_dim * 2, 1)
           self.purchase_predictor = nn.Linear(embedding_dim * 2, 1)
       
       def forward(self, user_ids, item_ids, task='rating'):
           user_emb = self.user_embedding(user_ids)
           item_emb = self.item_embedding(item_ids)
           combined = torch.cat([user_emb, item_emb], dim=1)
           
           if task == 'rating':
               return self.rating_predictor(combined)
           elif task == 'click':
               return torch.sigmoid(self.click_predictor(combined))
           elif task == 'purchase':
               return torch.sigmoid(self.purchase_predictor(combined))
   ```

2. **Causal Inference for Recommendations**
   ```python
   class CausalRecommender:
       def __init__(self, base_model, treatment_model):
           self.base_model = base_model
           self.treatment_model = treatment_model
       
       def estimate_treatment_effect(self, user_features, item_features):
           """Estimate causal effect of recommendation"""
           # Propensity score estimation
           propensity_scores = self.treatment_model.predict_proba(
               np.hstack([user_features, item_features])
           )
           
           # Inverse propensity weighting
           weights = 1.0 / propensity_scores
           
           # Causal effect estimation
           treated_outcome = self.base_model.predict(
               user_features, item_features, treatment=1
           )
           control_outcome = self.base_model.predict(
               user_features, item_features, treatment=0
           )
           
           causal_effect = treated_outcome - control_outcome
           return causal_effect * weights
   ```

3. **Federated Learning for Privacy-Preserving Recommendations**
   ```python
   class FederatedRecommender:
       def __init__(self, global_model):
           self.global_model = global_model
           self.client_models = {}
       
       def federated_averaging(self, client_updates):
           """Aggregate client model updates"""
           global_state = self.global_model.state_dict()
           
           for key in global_state.keys():
               # Average parameters across clients
               updates = [client_update[key] for client_update in client_updates]
               global_state[key] = torch.stack(updates).mean(0)
           
           self.global_model.load_state_dict(global_state)
       
       def train_federated_round(self, client_data):
           """Execute one round of federated training"""
           client_updates = []
           
           for client_id, data in client_data.items():
               # Send global model to client
               client_model = copy.deepcopy(self.global_model)
               
               # Train on client data
               client_model = self.train_client_model(client_model, data)
               
               # Collect model update
               client_updates.append(client_model.state_dict())
           
           # Aggregate updates
           self.federated_averaging(client_updates)
   ```

4. **Sequential and Session-Based Recommendations**
   ```python
   class SessionBasedGRU(nn.Module):
       def __init__(self, n_items, embedding_dim=64, hidden_dim=128):
           super().__init__()
           
           self.item_embedding = nn.Embedding(n_items, embedding_dim)
           self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
           self.output_layer = nn.Linear(hidden_dim, n_items)
       
       def forward(self, session_items, session_lengths):
           # Embed items
           embedded = self.item_embedding(session_items)
           
           # Pack sequences for efficient processing
           packed = nn.utils.rnn.pack_padded_sequence(
               embedded, session_lengths, batch_first=True, enforce_sorted=False
           )
           
           # GRU processing
           output, hidden = self.gru(packed)
           
           # Unpack and get final hidden state
           unpacked, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
           
           # Use last relevant output for each sequence
           batch_size = session_items.size(0)
           last_outputs = unpacked[range(batch_size), session_lengths - 1]
           
           # Predict next item
           scores = self.output_layer(last_outputs)
           return scores
   ```

### Novel Experiments
- **Cross-domain recommendations**: Transfer learning between different domains
- **Explainable recommendations**: Interpretable models with reason codes
- **Conversational recommendations**: Dialogue-based preference elicitation
- **Time-aware recommendations**: Modeling temporal dynamics and seasonality
- **Social recommendations**: Incorporating social network information

### Emerging Applications
```python
# Reinforcement Learning for recommendations
class ReinforcementLearningRecommender:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.target_network = copy.deepcopy(self.q_network)
        self.optimizer = torch.optim.Adam(self.q_network.parameters())
        
    def select_action(self, state, epsilon=0.1):
        """Epsilon-greedy action selection"""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-network using DQN"""
        with torch.no_grad():
            next_q_values = self.target_network(next_state)
            target = reward + (1 - done) * 0.99 * next_q_values.max()
        
        current_q = self.q_network(state)[action]
        loss = nn.MSELoss()(current_q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Knowledge Graph Enhanced Recommendations
class KnowledgeGraphRecommender:
    def __init__(self, kg_embeddings, entity_dim=64):
        self.kg_embeddings = kg_embeddings
        self.entity_dim = entity_dim
        
    def get_entity_embeddings(self, entities):
        """Get embeddings for entities from knowledge graph"""
        embeddings = []
        for entity in entities:
            if entity in self.kg_embeddings:
                embeddings.append(self.kg_embeddings[entity])
            else:
                # Random embedding for unknown entities
                embeddings.append(np.random.normal(0, 0.1, self.entity_dim))
        
        return np.array(embeddings)
    
    def compute_path_based_similarity(self, item1, item2):
        """Compute similarity based on knowledge graph paths"""
        # Use graph algorithms to find paths between items
        paths = self.find_paths(item1, item2, max_length=3)
        
        path_scores = []
        for path in paths:
            # Score path based on relation types and lengths
            path_score = 1.0 / len(path)  # Shorter paths are more important
            path_scores.append(path_score)
        
        return sum(path_scores)
```

### Industry Applications
- **E-commerce**: Product recommendations with inventory constraints
- **Streaming**: Content recommendations with engagement optimization
- **Social media**: Feed curation and friend suggestions
- **News**: Article recommendations with freshness and diversity
- **Finance**: Investment and insurance product recommendations
- **Healthcare**: Treatment and drug recommendations
- **Education**: Course and learning path recommendations

## 10. Portfolio Polish

### Documentation Structure
```
recommender_system/
├── README.md                           # This file
├── notebooks/
│   ├── 01_Data_Exploration.ipynb       # MovieLens/Amazon data analysis
│   ├── 02_Baseline_Models.ipynb        # Collaborative filtering baselines
│   ├── 03_Matrix_Factorization.ipynb   # SVD and advanced factorization
│   ├── 04_Deep_Learning.ipynb          # Neural collaborative filtering
│   ├── 05_Graph_Neural_Networks.ipynb  # GNN-based recommendations
│   └── 06_Evaluation_Study.ipynb       # Comprehensive evaluation
├── src/
│   ├── models/
│   │   ├── collaborative_filtering.py
│   │   ├── matrix_factorization.py
│   │   ├── neural_collaborative.py
│   │   ├── graph_recommender.py
│   │   ├── session_based.py
│   │   └── multi_task.py
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── data_loaders.py
│   │   ├── negative_sampling.py
│   │   └── feature_engineering.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── cold_start_eval.py
│   │   ├── fairness_eval.py
│   │   └── ab_testing.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── curriculum_learning.py
│   │   └── multi_task_trainer.py
│   ├── inference/
│   │   ├── recommendation_service.py
│   │   ├── similarity_service.py
│   │   └── explanation_service.py
│   ├── train.py
│   ├── recommend.py
│   └── evaluate.py
├── configs/
│   ├── collaborative_filtering.yaml
│   ├── neural_cf.yaml
│   ├── graph_recommender.yaml
│   └── training_configs/
├── api/
│   ├── fastapi_server.py
│   ├── recommendation_service.py
│   ├── streaming_service.py
│   └── requirements.txt
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── kubernetes/
│   ├── batch_pipeline/
│   │   ├── airflow_dag.py
│   │   └── beam_pipeline.py
│   ├── streaming/
│   │   ├── kafka_consumer.py
│   │   └── real_time_service.py
│   └── cloud_deployment/
├── evaluation/
│   ├── benchmark_scripts/
│   ├── cold_start_evaluation.py
│   ├── fairness_analysis.py
│   ├── ab_testing_framework.py
│   └── offline_evaluation.py
├── datasets/
│   ├── movielens_loader.py
│   ├── amazon_loader.py
│   ├── custom_dataset.py
│   └── data_validation.py
├── demo/
│   ├── gradio_app.py
│   ├── streamlit_dashboard.py
│   ├── recommendation_explainer.py
│   └── mobile_demo/
├── tests/
│   ├── test_models.py
│   ├── test_evaluation.py
│   ├── test_api.py
│   └── test_preprocessing.py
├── requirements.txt
├── setup.py
├── Makefile
└── .github/workflows/
```

### Visualization Requirements
- **User-item interaction heatmaps**: Visualization of rating patterns
- **Embedding space visualization**: t-SNE/UMAP plots of user/item embeddings
- **Recommendation performance dashboards**: Real-time metrics monitoring
- **A/B testing results**: Statistical significance and lift analysis
- **Cold start performance**: New user/item recommendation quality
- **Diversity and novelty analysis**: Recommendation catalog coverage
- **Temporal patterns**: User behavior and recommendation trends over time
- **Fairness analysis**: Performance across demographic groups

### Blog Post Template
1. **The Recommendation Revolution**: How ML transforms content discovery
2. **Dataset Deep-dive**: MovieLens vs real-world recommendation challenges
3. **Algorithm Evolution**: From collaborative filtering to deep learning
4. **Neural Approaches**: Understanding embeddings and implicit feedback
5. **Production Challenges**: Scalability, latency, and cold start problems
6. **Evaluation Beyond Accuracy**: Diversity, novelty, and fairness metrics
7. **Real-world Deployment**: Building recommendation systems at scale
8. **Future Frontiers**: Causal inference, federated learning, and explainability

### Demo Video Script
- 1 minute: Recommendation systems in daily life and business impact
- 1.5 minutes: Dataset exploration with user-item interaction patterns
- 2 minutes: Model progression from simple baselines to neural networks
- 2.5 minutes: Live recommendation demo with different algorithms
- 1.5 minutes: Evaluation metrics and A/B testing insights
- 1 minute: Cold start and fairness analysis
- 2 minutes: Production deployment and real-time recommendation API
- 1 minute: Future research directions and advanced techniques

### GitHub README Essentials
```markdown
# Advanced Recommender Systems with Deep Learning

![Recommendation Demo](assets/recommendation_demo.gif)

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Download MovieLens dataset
python src/data/download_movielens.py --size 100k

# Train neural collaborative filtering model
python src/train.py --config configs/neural_cf.yaml

# Generate recommendations
python src/recommend.py --model models/best_model.pt --user_id 123

# Launch demo
python demo/gradio_app.py
```

## 📊 Results
| Model | Dataset | RMSE | Precision@10 | Recall@10 | NDCG@10 |
|-------|---------|------|--------------|-----------|---------|
| User-CF | MovieLens-100K | 0.98 | 0.127 | 0.089 | 0.156 |
| Matrix Factorization | MovieLens-100K | 0.87 | 0.145 | 0.112 | 0.178 |
| Neural CF | MovieLens-100K | 0.82 | 0.168 | 0.134 | 0.203 |
| Graph Neural Network | MovieLens-100K | 0.79 | 0.182 | 0.149 | 0.221 |

## 🎯 Live Demo
Try the recommendation system: [Hugging Face Space](https://huggingface.co/spaces/username/recommender-demo)

## 🏢 Production API
```python
import requests

response = requests.post("http://api.example.com/recommend", json={
    "user_id": "123",
    "num_recommendations": 10
})
print(response.json())
```

## 📚 Citation
```bibtex
@article{advanced_recommender_2024,
  title={Advanced Recommender Systems: From Collaborative Filtering to Graph Neural Networks},
  author={Your Name},
  journal={ACM Transactions on Recommender Systems},
  year={2024}
}
```
```

### Performance Benchmarks
- **Recommendation accuracy**: RMSE, MAE across different model types
- **Ranking quality**: Precision, Recall, NDCG at various cutoff points
- **Inference latency**: Response time for real-time recommendations
- **Throughput**: Recommendations per second under load
- **Memory usage**: Model size and RAM requirements
- **Cold start performance**: Quality for new users and items
- **Diversity metrics**: Intra-list diversity and catalog coverage
- **Fairness analysis**: Performance across demographic groups
- **A/B testing results**: Business metric improvements and statistical significance