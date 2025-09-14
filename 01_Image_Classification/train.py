"""
Training script for image classification
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
from datetime import datetime
import wandb
from typing import Dict, Tuple

from data_loader import load_cifar10, MixupDataset, CutMixDataset
from models import get_model
from utils import AverageMeter, accuracy, save_checkpoint, load_checkpoint


class Trainer:
    """Main trainer class"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project="image-classification",
                name=config.get('experiment_name', 'cifar10'),
                config=config
            )
        
        # Load data
        self.train_loader, self.val_loader, self.test_loader = load_cifar10(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        # Initialize model
        self.model = get_model(
            config['model_name'],
            num_classes=config['num_classes'],
            pretrained=config.get('pretrained', True)
        ).to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.get('label_smoothing', 0.1)
        )
        
        # Optimizer
        self.optimizer = self._get_optimizer()
        
        # Scheduler
        self.scheduler = self._get_scheduler()
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.start_epoch = 0
        self.best_acc = 0.0
        
        # Resume from checkpoint if specified
        if config.get('resume'):
            self.load_checkpoint(config['resume'])
    
    def _get_optimizer(self):
        """Get optimizer based on config"""
        opt_name = self.config.get('optimizer', 'adamw').lower()
        lr = self.config['learning_rate']
        weight_decay = self.config.get('weight_decay', 0.01)
        
        if opt_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _get_scheduler(self):
        """Get learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=1e-6
            )
        elif scheduler_name == 'onecycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config['learning_rate'],
                epochs=self.config['epochs'],
                steps_per_epoch=len(self.train_loader)
            )
        else:
            return None
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Apply mixup/cutmix if enabled
            if self.config.get('use_mixup', False) and np.random.rand() < 0.5:
                images, targets_a, targets_b, lam = self._mixup_data(images, targets)
                
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    if self.config.get('use_mixup', False) and 'lam' in locals():
                        loss = lam * self.criterion(outputs, targets_a) + \
                               (1 - lam) * self.criterion(outputs, targets_b)
                    else:
                        loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.config.get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # Update metrics
            if not (self.config.get('use_mixup', False) and 'lam' in locals()):
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc@1': f'{top1.avg:.2f}',
                'Acc@5': f'{top5.avg:.2f}'
            })
            
            # Update scheduler if OneCycle
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
        
        return {
            'loss': losses.avg,
            'acc1': top1.avg,
            'acc5': top5.avg
        }
    
    def validate(self, loader=None) -> Dict:
        """Validate the model"""
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        with torch.no_grad():
            for images, targets in tqdm(loader, desc="Validating"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
        
        return {
            'loss': losses.avg,
            'acc1': top1.avg,
            'acc5': top5.avg
        }
    
    def _mixup_data(self, x, y, alpha=0.2):
        """Apply mixup augmentation"""
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.start_epoch, self.config['epochs']):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler and not isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # Log metrics
            print(f"\nEpoch {epoch}/{self.config['epochs']}:")
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc@1: {train_metrics['acc1']:.2f}%, "
                  f"Train Acc@5: {train_metrics['acc5']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc@1: {val_metrics['acc1']:.2f}%, "
                  f"Val Acc@5: {val_metrics['acc5']:.2f}%")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_acc1': train_metrics['acc1'],
                    'train_acc5': train_metrics['acc5'],
                    'val_loss': val_metrics['loss'],
                    'val_acc1': val_metrics['acc1'],
                    'val_acc5': val_metrics['acc5'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            is_best = val_metrics['acc1'] > self.best_acc
            if is_best:
                self.best_acc = val_metrics['acc1']
            
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_acc': self.best_acc,
                'config': self.config
            }, is_best, checkpoint_dir=self.config['checkpoint_dir'])
        
        # Final test evaluation
        print("\nEvaluating on test set...")
        test_metrics = self.validate(self.test_loader)
        print(f"Test Loss: {test_metrics['loss']:.4f}, "
              f"Test Acc@1: {test_metrics['acc1']:.2f}%, "
              f"Test Acc@5: {test_metrics['acc5']:.2f}%")
        
        if self.config.get('use_wandb', False):
            wandb.log({
                'test_loss': test_metrics['loss'],
                'test_acc1': test_metrics['acc1'],
                'test_acc5': test_metrics['acc5']
            })


def main():
    parser = argparse.ArgumentParser(description='Image Classification Training')
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override config with command line arguments
    config['model_name'] = args.model
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['resume'] = args.resume
    
    # Set defaults
    config.setdefault('num_classes', 10)
    config.setdefault('data_dir', './data')
    config.setdefault('checkpoint_dir', './checkpoints')
    config.setdefault('num_workers', 4)
    config.setdefault('use_amp', True)
    config.setdefault('use_wandb', False)
    config.setdefault('label_smoothing', 0.1)
    config.setdefault('weight_decay', 0.01)
    config.setdefault('gradient_clip', 1.0)
    config.setdefault('use_mixup', True)
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()