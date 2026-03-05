"""
Utility functions for machine translation
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import math
from typing import Tuple, List, Optional, Dict


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


def calculate_bleu(predictions: List[str], references: List[str]) -> Dict:
    """
    Calculate BLEU score using sacrebleu

    Args:
        predictions: List of predicted translations
        references: List of reference translations

    Returns:
        Dictionary with BLEU score and related metrics
    """
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        return {
            'bleu': bleu.score,
            'brevity_penalty': bleu.bp,
            'precisions': bleu.precisions
        }
    except ImportError:
        # Fallback: simple BLEU-like score
        return {'bleu': 0.0, 'brevity_penalty': 0.0, 'precisions': [0.0] * 4}


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

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def plot_training_curves(history: dict, save_path: Optional[str] = None):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # BLEU plot
    if 'val_bleu' in history:
        axes[1].plot(history['val_bleu'], label='Val BLEU', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('BLEU Score')
        axes[1].set_title('Validation BLEU Score')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def calculate_model_stats(model: nn.Module) -> dict:
    """Calculate model statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024 / 1024

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb
    }


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric):
        if self.mode == 'min':
            score = -val_metric
        else:
            score = val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop
