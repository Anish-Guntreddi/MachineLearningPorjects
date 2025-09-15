"""
Utility functions for Anomaly Detection
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, confusion_matrix
)
from typing import Dict, List, Tuple, Optional
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


def calculate_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate anomaly detection metrics
    
    Args:
        scores: Anomaly scores
        labels: True labels (0: normal, 1: anomaly)
        threshold: Threshold for classification
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # AUC scores
    if len(np.unique(labels)) > 1:
        metrics['auc_roc'] = roc_auc_score(labels, scores)
        metrics['auc_pr'] = average_precision_score(labels, scores)
    else:
        metrics['auc_roc'] = 0.5
        metrics['auc_pr'] = 0.5
    
    # Classification metrics with threshold
    if threshold is not None:
        predictions = (scores > threshold).astype(int)
        
        if len(np.unique(labels)) > 1:
            metrics['f1'] = f1_score(labels, predictions)
            metrics['precision'] = precision_score(labels, predictions, zero_division=0)
            metrics['recall'] = recall_score(labels, predictions, zero_division=0)
        else:
            metrics['f1'] = 0.0
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
    
    return metrics


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    save_path: Optional[str] = None
):
    """Plot ROC curve"""
    if len(np.unique(labels)) <= 1:
        print("Cannot plot ROC curve with single class")
        return
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_pr_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    save_path: Optional[str] = None
):
    """Plot Precision-Recall curve"""
    if len(np.unique(labels)) <= 1:
        print("Cannot plot PR curve with single class")
        return
    
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    auc_pr = average_precision_score(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {auc_pr:.3f})', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_anomaly_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float] = None,
    save_path: Optional[str] = None
):
    """Plot distribution of anomaly scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1] if np.any(labels == 1) else []
    
    ax1.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    if len(anomaly_scores) > 0:
        ax1.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
    
    if threshold is not None:
        ax1.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold={threshold:.3f}')
    
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Anomaly Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [normal_scores]
    labels_to_plot = ['Normal']
    
    if len(anomaly_scores) > 0:
        data_to_plot.append(anomaly_scores)
        labels_to_plot.append('Anomaly')
    
    ax2.boxplot(data_to_plot, labels=labels_to_plot)
    if threshold is not None:
        ax2.axhline(y=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold={threshold:.3f}')
    
    ax2.set_ylabel('Anomaly Score')
    ax2.set_title('Anomaly Scores by Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_aucs: List[float],
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
    
    # AUC plot
    ax2.plot(epochs, val_aucs, 'g-', label='Val AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title('Validation AUC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def visualize_reconstruction(
    model: torch.nn.Module,
    data: np.ndarray,
    device: torch.device,
    n_samples: int = 5,
    save_path: Optional[str] = None
):
    """Visualize original vs reconstructed data"""
    model.eval()
    
    n_samples = min(n_samples, len(data))
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 3 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(n_samples):
            sample = torch.tensor(data[i], dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get reconstruction
            if hasattr(model, 'forward'):
                output = model(sample)
                if isinstance(output, tuple):
                    reconstructed = output[0]
                else:
                    reconstructed = output
            else:
                reconstructed = sample
            
            reconstructed = reconstructed.cpu().numpy().squeeze()
            original = data[i]
            
            # Plot original
            if len(original.shape) == 1:
                axes[i, 0].plot(original)
                axes[i, 0].set_title(f'Original Sample {i+1}')
            else:
                axes[i, 0].imshow(original, cmap='viridis')
                axes[i, 0].set_title(f'Original Sample {i+1}')
            
            # Plot reconstruction
            if len(reconstructed.shape) == 1:
                axes[i, 1].plot(reconstructed)
                axes[i, 1].set_title(f'Reconstructed Sample {i+1}')
            else:
                axes[i, 1].imshow(reconstructed, cmap='viridis')
                axes[i, 1].set_title(f'Reconstructed Sample {i+1}')
            
            # Calculate reconstruction error
            error = np.mean((original - reconstructed) ** 2)
            axes[i, 1].text(0.02, 0.98, f'MSE: {error:.4f}',
                          transform=axes[i, 1].transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[str] = None
):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_latent_space(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: Optional[str] = None
):
    """Visualize latent space (for models with encoder)"""
    if not hasattr(model, 'encode'):
        print("Model does not have encode method")
        return
    
    model.eval()
    
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            data = batch['data'].to(device)
            label = batch['label']
            
            # Get latent representation
            if hasattr(model, 'encode'):
                z = model.encode(data)
                if isinstance(z, tuple):
                    z = z[0]  # For VAE, use mean
            else:
                continue
            
            latent_vectors.append(z.cpu().numpy())
            labels.append(label.numpy())
    
    if not latent_vectors:
        return
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Use PCA for visualization if dimension > 2
    if latent_vectors.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_vectors)
    else:
        latent_2d = latent_vectors
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    plt.scatter(latent_2d[normal_mask, 0], latent_2d[normal_mask, 1],
               c='blue', label='Normal', alpha=0.6, s=20)
    
    if np.any(anomaly_mask):
        plt.scatter(latent_2d[anomaly_mask, 0], latent_2d[anomaly_mask, 1],
                   c='red', label='Anomaly', alpha=0.6, s=20)
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def find_optimal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    metric: str = 'f1'
) -> float:
    """
    Find optimal threshold for anomaly detection
    
    Args:
        scores: Anomaly scores
        labels: True labels
        metric: Metric to optimize ('f1', 'precision', 'recall')
    
    Returns:
        Optimal threshold
    """
    thresholds = np.percentile(scores, np.linspace(50, 99, 50))
    best_score = 0
    best_threshold = thresholds[0]
    
    for threshold in thresholds:
        predictions = (scores > threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(labels, predictions, zero_division=0)
        elif metric == 'precision':
            score = precision_score(labels, predictions, zero_division=0)
        elif metric == 'recall':
            score = recall_score(labels, predictions, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold


def contamination_analysis(
    scores: np.ndarray,
    contamination_rates: List[float] = [0.01, 0.05, 0.1, 0.15, 0.2]
) -> Dict[int, float]:
    """
    Analyze thresholds for different contamination rates
    
    Args:
        scores: Anomaly scores
        contamination_rates: List of contamination rates to analyze
    
    Returns:
        Dictionary mapping contamination rate to threshold
    """
    thresholds = {}
    
    for rate in contamination_rates:
        percentile = 100 * (1 - rate)
        threshold = np.percentile(scores, percentile)
        thresholds[rate] = threshold
    
    return thresholds


if __name__ == "__main__":
    # Test utilities
    print("Testing anomaly detection utilities...")
    
    # Generate dummy data
    np.random.seed(42)
    n_normal = 900
    n_anomaly = 100
    
    normal_scores = np.random.normal(0, 1, n_normal)
    anomaly_scores = np.random.normal(3, 1, n_anomaly)
    
    scores = np.concatenate([normal_scores, anomaly_scores])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # Shuffle
    idx = np.random.permutation(len(scores))
    scores = scores[idx]
    labels = labels[idx]
    
    # Test metrics
    threshold = np.percentile(scores, 90)
    metrics = calculate_metrics(scores, labels, threshold)
    print(f"Metrics: {metrics}")
    
    # Test optimal threshold
    optimal_threshold = find_optimal_threshold(scores, labels, 'f1')
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Test contamination analysis
    contamination_thresholds = contamination_analysis(scores)
    print(f"Contamination thresholds: {contamination_thresholds}")
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3, mode='min')
    losses = [0.5, 0.45, 0.46, 0.47, 0.48]
    for i, loss in enumerate(losses):
        if early_stop(loss):
            print(f"Early stopping at iteration {i+1}")
            break