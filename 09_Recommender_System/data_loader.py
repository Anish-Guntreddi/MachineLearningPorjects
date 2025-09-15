"""
Data loading and preprocessing for Recommender Systems
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class RatingDataset(Dataset):
    """Dataset for rating prediction"""
    
    def __init__(
        self,
        ratings_df: pd.DataFrame,
        user_features: Optional[pd.DataFrame] = None,
        item_features: Optional[pd.DataFrame] = None,
        negative_sampling: bool = False,
        num_negatives: int = 4
    ):
        """
        Args:
            ratings_df: DataFrame with columns [user_id, item_id, rating, ...]
            user_features: Optional user features DataFrame
            item_features: Optional item features DataFrame
            negative_sampling: Whether to use negative sampling
            num_negatives: Number of negative samples per positive sample
        """
        self.ratings_df = ratings_df
        self.user_features = user_features
        self.item_features = item_features
        self.negative_sampling = negative_sampling
        self.num_negatives = num_negatives
        
        # Encode user and item IDs
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        self.ratings_df['user_idx'] = self.user_encoder.fit_transform(ratings_df['user_id'])
        self.ratings_df['item_idx'] = self.item_encoder.fit_transform(ratings_df['item_id'])
        
        self.num_users = len(self.user_encoder.classes_)
        self.num_items = len(self.item_encoder.classes_)
        
        # Create interaction matrix for negative sampling
        if negative_sampling:
            self._create_interaction_matrix()
            self._prepare_negative_samples()
    
    def _create_interaction_matrix(self):
        """Create sparse interaction matrix"""
        self.interaction_matrix = sp.dok_matrix(
            (self.num_users, self.num_items),
            dtype=np.float32
        )
        
        for _, row in self.ratings_df.iterrows():
            self.interaction_matrix[row['user_idx'], row['item_idx']] = 1
    
    def _prepare_negative_samples(self):
        """Prepare negative samples for training"""
        self.negative_samples = []
        
        for _, row in self.ratings_df.iterrows():
            user_idx = row['user_idx']
            negatives = []
            
            # Sample negative items
            while len(negatives) < self.num_negatives:
                neg_item = np.random.randint(self.num_items)
                if self.interaction_matrix[user_idx, neg_item] == 0:
                    negatives.append(neg_item)
            
            self.negative_samples.append(negatives)
    
    def __len__(self):
        if self.negative_sampling:
            return len(self.ratings_df) * (1 + self.num_negatives)
        return len(self.ratings_df)
    
    def __getitem__(self, idx):
        if self.negative_sampling:
            # Handle negative sampling
            pos_idx = idx // (1 + self.num_negatives)
            neg_idx = idx % (1 + self.num_negatives)
            
            row = self.ratings_df.iloc[pos_idx]
            
            if neg_idx == 0:
                # Positive sample
                return {
                    'user': torch.tensor(row['user_idx'], dtype=torch.long),
                    'item': torch.tensor(row['item_idx'], dtype=torch.long),
                    'rating': torch.tensor(row.get('rating', 1.0), dtype=torch.float),
                    'label': torch.tensor(1.0, dtype=torch.float)
                }
            else:
                # Negative sample
                neg_item = self.negative_samples[pos_idx][neg_idx - 1]
                return {
                    'user': torch.tensor(row['user_idx'], dtype=torch.long),
                    'item': torch.tensor(neg_item, dtype=torch.long),
                    'rating': torch.tensor(0.0, dtype=torch.float),
                    'label': torch.tensor(0.0, dtype=torch.float)
                }
        else:
            row = self.ratings_df.iloc[idx]
            return {
                'user': torch.tensor(row['user_idx'], dtype=torch.long),
                'item': torch.tensor(row['item_idx'], dtype=torch.long),
                'rating': torch.tensor(row.get('rating', 1.0), dtype=torch.float)
            }


class SequentialDataset(Dataset):
    """Dataset for sequential recommendation"""
    
    def __init__(
        self,
        sequences_df: pd.DataFrame,
        max_seq_length: int = 50,
        item_encoder: Optional[LabelEncoder] = None
    ):
        """
        Args:
            sequences_df: DataFrame with user sequences
            max_seq_length: Maximum sequence length
            item_encoder: Pre-fitted item encoder
        """
        self.sequences_df = sequences_df
        self.max_seq_length = max_seq_length
        
        # Encode items
        if item_encoder is None:
            self.item_encoder = LabelEncoder()
            all_items = []
            for seq in sequences_df['item_sequence']:
                all_items.extend(seq)
            self.item_encoder.fit(all_items)
        else:
            self.item_encoder = item_encoder
        
        self.num_items = len(self.item_encoder.classes_)
        
        # Process sequences
        self.sequences = []
        for _, row in sequences_df.iterrows():
            seq = row['item_sequence']
            if len(seq) > 1:
                encoded_seq = self.item_encoder.transform(seq)
                self.sequences.append({
                    'user_id': row['user_id'],
                    'sequence': encoded_seq
                })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        data = self.sequences[idx]
        sequence = data['sequence']
        
        # Split into input and target
        if len(sequence) > self.max_seq_length + 1:
            # Truncate sequence
            start_idx = np.random.randint(0, len(sequence) - self.max_seq_length)
            sequence = sequence[start_idx:start_idx + self.max_seq_length + 1]
        
        input_seq = sequence[:-1]
        target = sequence[-1]
        
        # Pad sequence
        if len(input_seq) < self.max_seq_length:
            padding = [0] * (self.max_seq_length - len(input_seq))
            input_seq = padding + list(input_seq)
            
        return {
            'input_seq': torch.tensor(input_seq, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
            'seq_len': torch.tensor(len(sequence) - 1, dtype=torch.long)
        }


class MovieLensDataset:
    """MovieLens dataset loader"""
    
    def __init__(
        self,
        dataset_size: str = '100k',  # '100k', '1m', '20m'
        data_path: str = './data',
        min_rating: float = 4.0,
        implicit: bool = False
    ):
        """
        Args:
            dataset_size: Size of MovieLens dataset
            data_path: Path to data directory
            min_rating: Minimum rating for implicit feedback
            implicit: Whether to convert to implicit feedback
        """
        self.dataset_size = dataset_size
        self.data_path = Path(data_path)
        self.min_rating = min_rating
        self.implicit = implicit
        
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None
        
        self._load_data()
    
    def _load_data(self):
        """Load MovieLens data"""
        if self.dataset_size == '100k':
            self._load_ml100k()
        elif self.dataset_size == '1m':
            self._load_ml1m()
        elif self.dataset_size == '20m':
            self._load_ml20m()
        else:
            raise ValueError(f"Unknown dataset size: {self.dataset_size}")
        
        # Convert to implicit if specified
        if self.implicit:
            self.ratings_df['rating'] = (self.ratings_df['rating'] >= self.min_rating).astype(float)
    
    def _load_ml100k(self):
        """Load MovieLens 100K dataset"""
        # Download if not exists
        import requests
        import zipfile
        
        data_dir = self.data_path / 'ml-100k'
        if not data_dir.exists():
            url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
            zip_path = self.data_path / 'ml-100k.zip'
            
            # Download
            response = requests.get(url)
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_path)
        
        # Load ratings
        self.ratings_df = pd.read_csv(
            data_dir / 'u.data',
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )
        
        # Load movies
        self.movies_df = pd.read_csv(
            data_dir / 'u.item',
            sep='|',
            encoding='latin-1',
            names=['item_id', 'title', 'release_date', 'video_release_date',
                  'imdb_url'] + [f'genre_{i}' for i in range(19)],
            usecols=['item_id', 'title']
        )
        
        # Load users
        self.users_df = pd.read_csv(
            data_dir / 'u.user',
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
        )
    
    def _load_ml1m(self):
        """Load MovieLens 1M dataset"""
        # Similar implementation for 1M dataset
        pass
    
    def _load_ml20m(self):
        """Load MovieLens 20M dataset"""
        # Similar implementation for 20M dataset
        pass
    
    def get_train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        temporal_split: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets
        
        Args:
            test_size: Test set size
            random_state: Random seed
            temporal_split: Whether to use temporal split
        
        Returns:
            train_df, test_df
        """
        if temporal_split:
            # Sort by timestamp and split
            sorted_df = self.ratings_df.sort_values('timestamp')
            split_idx = int(len(sorted_df) * (1 - test_size))
            train_df = sorted_df.iloc[:split_idx]
            test_df = sorted_df.iloc[split_idx:]
        else:
            # Random split
            train_df, test_df = train_test_split(
                self.ratings_df,
                test_size=test_size,
                random_state=random_state
            )
        
        return train_df, test_df


class RecommenderDataModule:
    """Data module for recommender systems"""
    
    def __init__(
        self,
        dataset_name: str = 'movielens',
        data_path: str = './data',
        batch_size: int = 256,
        num_workers: int = 4,
        negative_sampling: bool = True,
        num_negatives: int = 4,
        test_size: float = 0.2,
        val_size: float = 0.1
    ):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.negative_sampling = negative_sampling
        self.num_negatives = num_negatives
        self.test_size = test_size
        self.val_size = val_size
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup datasets"""
        if self.dataset_name == 'movielens':
            # Load MovieLens data
            ml_data = MovieLensDataset(
                dataset_size='100k',
                data_path=self.data_path
            )
            
            # Split data
            train_val_df, test_df = ml_data.get_train_test_split(
                test_size=self.test_size
            )
            
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=self.val_size / (1 - self.test_size)
            )
            
            # Create datasets
            self.train_dataset = RatingDataset(
                train_df,
                negative_sampling=self.negative_sampling,
                num_negatives=self.num_negatives
            )
            
            self.val_dataset = RatingDataset(
                val_df,
                negative_sampling=False
            )
            
            self.test_dataset = RatingDataset(
                test_df,
                negative_sampling=False
            )
            
            # Store metadata
            self.num_users = self.train_dataset.num_users
            self.num_items = self.train_dataset.num_items
            self.user_encoder = self.train_dataset.user_encoder
            self.item_encoder = self.train_dataset.item_encoder
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def create_user_item_matrix(
    ratings_df: pd.DataFrame,
    user_col: str = 'user_id',
    item_col: str = 'item_id',
    rating_col: str = 'rating'
) -> sp.csr_matrix:
    """
    Create user-item interaction matrix
    
    Args:
        ratings_df: Ratings DataFrame
        user_col: User column name
        item_col: Item column name
        rating_col: Rating column name
    
    Returns:
        Sparse user-item matrix
    """
    # Encode users and items
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    user_idx = user_encoder.fit_transform(ratings_df[user_col])
    item_idx = item_encoder.fit_transform(ratings_df[item_col])
    ratings = ratings_df[rating_col].values
    
    # Create sparse matrix
    matrix = sp.csr_matrix(
        (ratings, (user_idx, item_idx)),
        shape=(len(user_encoder.classes_), len(item_encoder.classes_))
    )
    
    return matrix, user_encoder, item_encoder


def generate_negative_samples(
    user_item_matrix: sp.csr_matrix,
    num_negatives: int = 4
) -> List[Tuple[int, int]]:
    """
    Generate negative samples for training
    
    Args:
        user_item_matrix: User-item interaction matrix
        num_negatives: Number of negatives per positive
    
    Returns:
        List of (user, item) negative pairs
    """
    num_users, num_items = user_item_matrix.shape
    negative_samples = []
    
    for user in range(num_users):
        # Get items user has interacted with
        interacted_items = set(user_item_matrix[user].indices)
        
        # Sample negative items
        negatives = []
        while len(negatives) < num_negatives:
            item = np.random.randint(num_items)
            if item not in interacted_items:
                negatives.append(item)
                negative_samples.append((user, item))
    
    return negative_samples


if __name__ == "__main__":
    # Test data loading
    print("Testing recommender system data loading...")
    
    # Test MovieLens dataset
    ml_data = MovieLensDataset(
        dataset_size='100k',
        data_path='./data'
    )
    
    print(f"Number of ratings: {len(ml_data.ratings_df)}")
    print(f"Number of users: {ml_data.ratings_df['user_id'].nunique()}")
    print(f"Number of items: {ml_data.ratings_df['item_id'].nunique()}")
    
    # Test data module
    dm = RecommenderDataModule(
        dataset_name='movielens',
        batch_size=32,
        negative_sampling=True
    )
    
    dm.setup()
    print(f"Train dataset size: {len(dm.train_dataset)}")
    print(f"Val dataset size: {len(dm.val_dataset)}")
    print(f"Test dataset size: {len(dm.test_dataset)}")
    
    # Test batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"User shape: {batch['user'].shape}")
    print(f"Item shape: {batch['item'].shape}")