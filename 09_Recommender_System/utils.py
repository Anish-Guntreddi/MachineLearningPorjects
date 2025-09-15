"""
Utility functions for Recommender Systems
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, ndcg_score
from typing import Dict, List, Tuple, Optional
import scipy.sparse as sp
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, mode: str = 'min', delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.is_better = lambda x, y: x < y - delta
        else:
            self.is_better = lambda x, y: x > y + delta
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def evaluate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    users: np.ndarray,
    items: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate recommendation metrics
    
    Args:
        predictions: Predicted ratings/scores
        targets: True ratings/scores
        users: User IDs
        items: Item IDs
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Overall metrics
    metrics['rmse'] = np.sqrt(mean_squared_error(targets, predictions))
    metrics['mae'] = mean_absolute_error(targets, predictions)
    
    # Per-user metrics
    user_rmse = []
    user_mae = []
    
    for user in np.unique(users):
        user_mask = users == user
        user_preds = predictions[user_mask]
        user_targets = targets[user_mask]
        
        if len(user_preds) > 0:
            user_rmse.append(np.sqrt(mean_squared_error(user_targets, user_preds)))
            user_mae.append(mean_absolute_error(user_targets, user_preds))
    
    metrics['avg_user_rmse'] = np.mean(user_rmse)
    metrics['avg_user_mae'] = np.mean(user_mae)
    
    return metrics


def calculate_ranking_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    k_list: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Calculate ranking metrics (Precision@K, Recall@K, NDCG@K)
    
    Args:
        predictions: Predicted scores (user x item)
        targets: True interactions (user x item)
        k_list: List of K values
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    num_users = predictions.shape[0]
    
    for k in k_list:
        precision_list = []
        recall_list = []
        ndcg_list = []
        
        for user_idx in range(num_users):
            # Get top-k predictions
            top_k_items = np.argsort(predictions[user_idx])[-k:][::-1]
            
            # Get true positives
            true_items = np.where(targets[user_idx] > 0)[0]
            
            if len(true_items) == 0:
                continue
            
            # Calculate metrics
            hits = np.intersect1d(top_k_items, true_items)
            
            precision = len(hits) / k
            recall = len(hits) / len(true_items)
            
            precision_list.append(precision)
            recall_list.append(recall)
            
            # Calculate NDCG
            if len(hits) > 0:
                ndcg = ndcg_score([targets[user_idx][top_k_items]], 
                                 [predictions[user_idx][top_k_items]], k=k)
                ndcg_list.append(ndcg)
        
        metrics[f'precision@{k}'] = np.mean(precision_list)
        metrics[f'recall@{k}'] = np.mean(recall_list)
        metrics[f'ndcg@{k}'] = np.mean(ndcg_list) if ndcg_list else 0.0
    
    return metrics


def hit_rate_at_k(predictions: np.ndarray, targets: np.ndarray, k: int = 10) -> float:
    """
    Calculate hit rate at K
    
    Args:
        predictions: Predicted scores
        targets: True interactions
        k: Number of recommendations
    
    Returns:
        Hit rate
    """
    hits = 0
    total = 0
    
    for user_idx in range(len(predictions)):
        top_k = np.argsort(predictions[user_idx])[-k:][::-1]
        true_items = np.where(targets[user_idx] > 0)[0]
        
        if len(true_items) > 0:
            if len(np.intersect1d(top_k, true_items)) > 0:
                hits += 1
            total += 1
    
    return hits / total if total > 0 else 0.0


def coverage(predictions: np.ndarray, num_items: int, k: int = 10) -> float:
    """
    Calculate item coverage
    
    Args:
        predictions: Predicted scores (user x item)
        num_items: Total number of items
        k: Number of recommendations per user
    
    Returns:
        Coverage ratio
    """
    recommended_items = set()
    
    for user_idx in range(len(predictions)):
        top_k = np.argsort(predictions[user_idx])[-k:][::-1]
        recommended_items.update(top_k)
    
    return len(recommended_items) / num_items


def diversity(predictions: np.ndarray, item_features: Optional[np.ndarray] = None, k: int = 10) -> float:
    """
    Calculate recommendation diversity
    
    Args:
        predictions: Predicted scores (user x item)
        item_features: Item feature matrix (optional)
        k: Number of recommendations
    
    Returns:
        Average diversity score
    """
    diversity_scores = []
    
    for user_idx in range(len(predictions)):
        top_k = np.argsort(predictions[user_idx])[-k:][::-1]
        
        if item_features is not None:
            # Calculate pairwise distances
            features = item_features[top_k]
            distances = []
            
            for i in range(len(top_k)):
                for j in range(i + 1, len(top_k)):
                    dist = np.linalg.norm(features[i] - features[j])
                    distances.append(dist)
            
            if distances:
                diversity_scores.append(np.mean(distances))
        else:
            # Use prediction scores as proxy for diversity
            scores = predictions[user_idx][top_k]
            diversity_scores.append(np.std(scores))
    
    return np.mean(diversity_scores) if diversity_scores else 0.0


def novelty(predictions: np.ndarray, item_popularity: np.ndarray, k: int = 10) -> float:
    """
    Calculate recommendation novelty
    
    Args:
        predictions: Predicted scores (user x item)
        item_popularity: Item popularity scores
        k: Number of recommendations
    
    Returns:
        Average novelty score
    """
    novelty_scores = []
    
    for user_idx in range(len(predictions)):
        top_k = np.argsort(predictions[user_idx])[-k:][::-1]
        
        # Calculate novelty as inverse of popularity
        item_novelty = -np.log2(item_popularity[top_k] + 1e-10)
        novelty_scores.append(np.mean(item_novelty))
    
    return np.mean(novelty_scores)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: List[float],
    val_metrics: List[float],
    metric_name: str = 'RMSE',
    save_path: Optional[str] = None
):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Metric plot
    ax2.plot(epochs, train_metrics, 'b-', label=f'Train {metric_name}')
    ax2.plot(epochs, val_metrics, 'r-', label=f'Val {metric_name}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(metric_name)
    ax2.set_title(f'Training and Validation {metric_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def save_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_rmses: List[float],
    val_maes: List[float],
    save_path: Optional[str] = None
):
    """Save comprehensive training curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RMSE plot
    ax2.plot(epochs, val_rmses, 'g-', label='Val RMSE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Validation RMSE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # MAE plot
    ax3.plot(epochs, val_maes, 'm-', label='Val MAE')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MAE')
    ax3.set_title('Validation MAE')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Combined metrics
    ax4_twin = ax4.twinx()
    ax4.plot(epochs, val_rmses, 'g-', label='RMSE')
    ax4_twin.plot(epochs, val_maes, 'm-', label='MAE')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('RMSE', color='g')
    ax4_twin.set_ylabel('MAE', color='m')
    ax4.set_title('All Validation Metrics')
    ax4.tick_params(axis='y', labelcolor='g')
    ax4_twin.tick_params(axis='y', labelcolor='m')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_rating_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[str] = None
):
    """Plot rating distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # True ratings distribution
    ax1.hist(targets, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Count')
    ax1.set_title('True Rating Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Predicted vs True scatter plot
    ax2.scatter(targets, predictions, alpha=0.5, s=1)
    ax2.plot([targets.min(), targets.max()], 
             [targets.min(), targets.max()], 
             'r--', linewidth=2)
    ax2.set_xlabel('True Rating')
    ax2.set_ylabel('Predicted Rating')
    ax2.set_title('Predicted vs True Ratings')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_user_item_matrix(
    matrix: sp.spmatrix,
    max_users: int = 100,
    max_items: int = 100,
    save_path: Optional[str] = None
):
    """Plot user-item interaction matrix"""
    # Convert to dense for visualization (subset)
    dense_matrix = matrix[:max_users, :max_items].toarray()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(dense_matrix, cmap='YlOrRd', cbar_kws={'label': 'Rating'})
    plt.xlabel('Items')
    plt.ylabel('Users')
    plt.title('User-Item Interaction Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def calculate_sparsity(matrix: sp.spmatrix) -> float:
    """Calculate sparsity of interaction matrix"""
    total_elements = matrix.shape[0] * matrix.shape[1]
    non_zero_elements = matrix.nnz
    sparsity = 1 - (non_zero_elements / total_elements)
    return sparsity


def get_popular_items(
    interaction_matrix: sp.spmatrix,
    k: int = 10
) -> List[int]:
    """Get most popular items"""
    item_popularity = np.array(interaction_matrix.sum(axis=0)).flatten()
    popular_items = np.argsort(item_popularity)[-k:][::-1]
    return popular_items.tolist()


def get_user_statistics(
    interaction_matrix: sp.spmatrix
) -> Dict[str, float]:
    """Get user interaction statistics"""
    user_interactions = np.array(interaction_matrix.sum(axis=1)).flatten()
    
    stats = {
        'avg_interactions': np.mean(user_interactions),
        'std_interactions': np.std(user_interactions),
        'min_interactions': np.min(user_interactions),
        'max_interactions': np.max(user_interactions),
        'median_interactions': np.median(user_interactions)
    }
    
    return stats


def get_item_statistics(
    interaction_matrix: sp.spmatrix
) -> Dict[str, float]:
    """Get item interaction statistics"""
    item_interactions = np.array(interaction_matrix.sum(axis=0)).flatten()
    
    stats = {
        'avg_interactions': np.mean(item_interactions),
        'std_interactions': np.std(item_interactions),
        'min_interactions': np.min(item_interactions),
        'max_interactions': np.max(item_interactions),
        'median_interactions': np.median(item_interactions)
    }
    
    return stats


if __name__ == "__main__":
    # Test utilities
    print("Testing recommender system utilities...")
    
    # Generate dummy data
    num_users = 100
    num_items = 50
    
    predictions = np.random.rand(num_users, num_items) * 5
    targets = np.random.randint(0, 2, (num_users, num_items))
    
    # Test ranking metrics
    ranking_metrics = calculate_ranking_metrics(predictions, targets)
    print(f"Ranking metrics: {ranking_metrics}")
    
    # Test hit rate
    hr = hit_rate_at_k(predictions, targets, k=10)
    print(f"Hit Rate@10: {hr:.4f}")
    
    # Test coverage
    cov = coverage(predictions, num_items, k=10)
    print(f"Coverage: {cov:.4f}")
    
    # Test diversity
    div = diversity(predictions, k=10)
    print(f"Diversity: {div:.4f}")
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3, mode='min')
    scores = [0.5, 0.45, 0.46, 0.47, 0.48]
    for i, score in enumerate(scores):
        if early_stop(score):
            print(f"Early stopping at iteration {i+1}")
            break