"""
Utility functions for text classification
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from wordcloud import WordCloud
import os
import shutil
from typing import Dict, List, Optional, Tuple


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str = './checkpoints'):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save(state, filename)
    
    if is_best:
        best_filename = os.path.join(checkpoint_dir, 'best_model.pth')
        shutil.copyfile(filename, best_filename)


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: Optional = None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True
):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the plot
        normalize: Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None
):
    """
    Plot training history
    
    Args:
        history: Dictionary containing training metrics
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history.get('train_loss', []), label='Train Loss')
    axes[0].plot(history.get('val_loss', []), label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history.get('train_acc', []), label='Train Acc')
    axes[1].plot(history.get('val_acc', []), label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> str:
    """
    Generate classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the report
    
    Returns:
        Classification report as string
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report


def create_word_cloud(
    texts: List[str],
    save_path: Optional[str] = None,
    max_words: int = 100
):
    """
    Create word cloud from texts
    
    Args:
        texts: List of text strings
        save_path: Optional path to save the plot
        max_words: Maximum number of words to display
    """
    # Combine all texts
    combined_text = ' '.join(texts)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        max_words=max_words,
        background_color='white',
        colormap='viridis'
    ).generate(combined_text)
    
    # Plot
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def analyze_predictions(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    num_examples: int = 5
) -> Dict:
    """
    Analyze predictions and find misclassified examples
    
    Args:
        texts: List of input texts
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        num_examples: Number of examples to return
    
    Returns:
        Dictionary with analysis results
    """
    misclassified_mask = y_true != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    # Get misclassified examples
    misclassified_examples = []
    for idx in misclassified_indices[:num_examples]:
        example = {
            'text': texts[idx],
            'true_label': class_names[y_true[idx]],
            'predicted_label': class_names[y_pred[idx]],
            'index': idx
        }
        misclassified_examples.append(example)
    
    # Calculate per-class accuracy
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == y_true[class_mask]).mean()
            per_class_accuracy[class_name] = float(class_acc)
    
    return {
        'misclassified_examples': misclassified_examples,
        'per_class_accuracy': per_class_accuracy,
        'total_misclassified': int(misclassified_mask.sum()),
        'total_samples': len(y_true),
        'error_rate': float(misclassified_mask.mean())
    }


def export_to_onnx(
    model: nn.Module,
    tokenizer,
    output_path: str,
    sample_text: str = "This is a sample text for export.",
    max_length: int = 512
):
    """
    Export model to ONNX format
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer for the model
        output_path: Path to save ONNX model
        sample_text: Sample text for tracing
        max_length: Maximum sequence length
    """
    model.eval()
    
    # Prepare sample input
    inputs = tokenizer(
        sample_text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Export
    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask']),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")


def calculate_model_stats(model: nn.Module) -> Dict:
    """
    Calculate model statistics
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': model_size_mb
    }


if __name__ == "__main__":
    # Test utilities
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    class_names = ['Negative', 'Positive']
    
    # Test confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Test classification report
    report = generate_classification_report(y_true, y_pred, class_names)
    print(report)