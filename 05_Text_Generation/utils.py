"""
Utility functions for text generation
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import math
from typing import Tuple, List, Optional


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


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss"""
    return math.exp(min(loss, 100))


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

    # Perplexity plot
    train_ppl = [calculate_perplexity(l) for l in history['train_loss']]
    val_ppl = [calculate_perplexity(l) for l in history['val_loss']]
    axes[1].plot(train_ppl, label='Train Perplexity')
    axes[1].plot(val_ppl, label='Val Perplexity')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Training and Validation Perplexity')
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
