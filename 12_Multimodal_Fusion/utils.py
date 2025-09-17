"""
Utility functions for Multimodal Fusion
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional, Any
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
    predictions: np.ndarray,
    labels: np.ndarray,
    task: str = 'classification'
) -> Dict[str, float]:
    """
    Calculate metrics for multimodal tasks
    
    Args:
        predictions: Model predictions
        labels: True labels
        task: Task type
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    if task in ['classification', 'vqa']:
        # Classification metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        
        # Handle binary and multiclass
        average = 'binary' if len(np.unique(labels)) == 2 else 'macro'
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=average, zero_division=0
        )
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
    elif task == 'matching':
        # Binary matching metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
    elif task == 'retrieval':
        # Retrieval metrics
        metrics['recall@1'] = recall_at_k(predictions, labels, k=1)
        metrics['recall@5'] = recall_at_k(predictions, labels, k=5)
        metrics['recall@10'] = recall_at_k(predictions, labels, k=10)
        
    return metrics


def recall_at_k(predictions: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Calculate Recall@K for retrieval tasks"""
    if len(predictions.shape) == 1:
        # Single prediction per sample
        return float(predictions == labels)
    
    # Multiple predictions per sample (ranked)
    correct = 0
    for pred, label in zip(predictions, labels):
        if label in pred[:k]:
            correct += 1
    
    return correct / len(labels)


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None
):
    """Plot training curves"""
    epochs = range(1, len(train_losses) + 1)
    
    # Determine number of subplots
    n_plots = 1
    if train_metrics or val_metrics:
        n_plots = 2
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot losses
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot metrics if available
    if n_plots > 1 and (train_metrics or val_metrics):
        metric_names = set()
        if train_metrics:
            metric_names.update(train_metrics.keys())
        if val_metrics:
            metric_names.update(val_metrics.keys())
        
        for metric in metric_names:
            if train_metrics and metric in train_metrics:
                axes[1].plot(epochs[:len(train_metrics[metric])], 
                           train_metrics[metric], 
                           label=f'Train {metric}')
            if val_metrics and metric in val_metrics:
                axes[1].plot(epochs[:len(val_metrics[metric])], 
                           val_metrics[metric], 
                           label=f'Val {metric}')
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric Value')
        axes[1].set_title('Training Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def visualize_attention_weights(
    attention_weights: torch.Tensor,
    source_tokens: Optional[List[str]] = None,
    target_tokens: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """Visualize cross-modal attention weights"""
    # Convert to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Handle multi-head attention
    if len(attention_weights.shape) == 4:
        # Average over heads
        attention_weights = attention_weights.mean(axis=1)
    
    # Take first sample if batch
    if len(attention_weights.shape) == 3:
        attention_weights = attention_weights[0]
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(attention_weights, cmap='Blues', cbar_kws={'label': 'Attention Weight'})
    
    # Set labels if provided
    if source_tokens:
        plt.yticks(range(len(source_tokens)), source_tokens, rotation=0)
    if target_tokens:
        plt.xticks(range(len(target_tokens)), target_tokens, rotation=45, ha='right')
    
    plt.title('Cross-Modal Attention Weights')
    plt.xlabel('Target Modality')
    plt.ylabel('Source Modality')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_modality_contributions(
    contributions: Dict[str, float],
    save_path: Optional[str] = None
):
    """Plot contribution of each modality"""
    modalities = list(contributions.keys())
    values = list(contributions.values())
    
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(modalities)))
    bars = plt.bar(modalities, values, color=colors)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.xlabel('Modality')
    plt.ylabel('Contribution Score')
    plt.title('Modality Contributions to Prediction')
    plt.ylim(0, max(values) * 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def visualize_fusion_process(
    features: Dict[str, np.ndarray],
    fused_features: np.ndarray,
    save_path: Optional[str] = None
):
    """Visualize the fusion process"""
    n_modalities = len(features)
    
    fig, axes = plt.subplots(1, n_modalities + 1, figsize=(4 * (n_modalities + 1), 4))
    
    # Plot individual modality features
    for i, (modality, feat) in enumerate(features.items()):
        if len(feat.shape) == 1:
            feat = feat.reshape(-1, 1)
        
        im = axes[i].imshow(feat[:50, :50], cmap='viridis', aspect='auto')
        axes[i].set_title(f'{modality.capitalize()} Features')
        axes[i].set_xlabel('Feature Dim')
        axes[i].set_ylabel('Sample Dim')
        plt.colorbar(im, ax=axes[i])
    
    # Plot fused features
    if len(fused_features.shape) == 1:
        fused_features = fused_features.reshape(-1, 1)
    
    im = axes[-1].imshow(fused_features[:50, :50], cmap='plasma', aspect='auto')
    axes[-1].set_title('Fused Features')
    axes[-1].set_xlabel('Feature Dim')
    axes[-1].set_ylabel('Sample Dim')
    plt.colorbar(im, ax=axes[-1])
    
    plt.suptitle('Multimodal Feature Fusion Visualization')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_retrieval_results(
    query_features: np.ndarray,
    retrieved_features: np.ndarray,
    similarities: np.ndarray,
    query_label: str = 'Query',
    retrieved_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """Visualize retrieval results"""
    n_retrieved = len(retrieved_features)
    
    fig, axes = plt.subplots(2, n_retrieved + 1, figsize=(3 * (n_retrieved + 1), 6))
    
    # Plot query
    if len(query_features.shape) == 1:
        query_features = query_features.reshape(-1, 1)
    
    axes[0, 0].imshow(query_features[:50, :], cmap='coolwarm', aspect='auto')
    axes[0, 0].set_title(query_label)
    axes[0, 0].set_ylabel('Feature Dim')
    axes[0, 0].axis('off')
    
    axes[1, 0].axis('off')
    
    # Plot retrieved items
    for i in range(n_retrieved):
        feat = retrieved_features[i]
        if len(feat.shape) == 1:
            feat = feat.reshape(-1, 1)
        
        axes[0, i+1].imshow(feat[:50, :], cmap='coolwarm', aspect='auto')
        
        label = f'Retrieved {i+1}'
        if retrieved_labels:
            label = retrieved_labels[i]
        
        axes[0, i+1].set_title(label)
        axes[0, i+1].axis('off')
        
        # Plot similarity score
        axes[1, i+1].bar([0], [similarities[i]], color='green')
        axes[1, i+1].set_ylim(0, 1)
        axes[1, i+1].set_ylabel('Similarity')
        axes[1, i+1].set_xticks([])
        axes[1, i+1].set_title(f'Score: {similarities[i]:.3f}')
    
    plt.suptitle('Cross-Modal Retrieval Results')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def compute_modality_importance(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    modalities: List[str]
) -> Dict[str, float]:
    """
    Compute importance of each modality using ablation
    
    Args:
        model: Trained multimodal model
        data_loader: Data loader
        device: Device to use
        modalities: List of modality names
    
    Returns:
        Dictionary of modality importance scores
    """
    model.eval()
    
    # Baseline performance with all modalities
    baseline_acc = evaluate_model(model, data_loader, device)
    
    importance_scores = {}
    
    # Ablate each modality
    for modality in modalities:
        # Create data loader with ablated modality
        ablated_acc = evaluate_model_ablated(
            model, data_loader, device, ablated_modality=modality
        )
        
        # Importance = drop in performance
        importance_scores[modality] = baseline_acc - ablated_acc
    
    # Normalize scores
    total_importance = sum(importance_scores.values())
    if total_importance > 0:
        importance_scores = {
            k: v / total_importance for k, v in importance_scores.items()
        }
    
    return importance_scores


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> float:
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return correct / total if total > 0 else 0


def evaluate_model_ablated(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    ablated_modality: str
) -> float:
    """Evaluate model with one modality ablated"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = {}
            for k, v in batch.items():
                if k == 'label':
                    labels = v.to(device)
                elif ablated_modality not in k:
                    inputs[k] = v.to(device) if isinstance(v, torch.Tensor) else v
            
            if not inputs:
                continue
            
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return correct / total if total > 0 else 0


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: str
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    filepath: str
) -> Dict[str, Any]:
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def generate_synthetic_multimodal_data(
    n_samples: int = 1000,
    image_dim: int = 2048,
    text_dim: int = 768,
    audio_dim: int = 128,
    n_classes: int = 10
) -> Dict[str, np.ndarray]:
    """Generate synthetic multimodal data for testing"""
    np.random.seed(42)
    
    data = {
        'image': np.random.randn(n_samples, image_dim).astype(np.float32),
        'text': np.random.randn(n_samples, text_dim).astype(np.float32),
        'audio': np.random.randn(n_samples, audio_dim).astype(np.float32),
        'labels': np.random.randint(0, n_classes, n_samples)
    }
    
    return data


if __name__ == "__main__":
    # Test utilities
    print("Testing multimodal fusion utilities...")
    
    # Generate synthetic data
    data = generate_synthetic_multimodal_data(n_samples=100)
    
    # Test metrics
    predictions = np.random.randint(0, 10, 100)
    labels = data['labels']
    
    metrics = calculate_metrics(predictions, labels, task='classification')
    print(f"Classification metrics: {metrics}")
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3, mode='max')
    scores = [0.5, 0.6, 0.58, 0.57, 0.56]
    for i, score in enumerate(scores):
        if early_stop(score):
            print(f"Early stopping triggered at iteration {i+1}")
            break
    
    # Test visualization
    contributions = {
        'image': 0.45,
        'text': 0.35,
        'audio': 0.20
    }
    plot_modality_contributions(contributions)
    
    print("Utilities test completed!")