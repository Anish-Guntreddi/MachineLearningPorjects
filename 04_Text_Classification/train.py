"""
Training script for text classification
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
from datetime import datetime
import wandb
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from data_loader import create_data_loaders
from models import get_model
from utils import AverageMeter, save_checkpoint, load_checkpoint, EarlyStopping


class TextClassificationTrainer:
    """Text classification trainer"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project="text-classification",
                name=config.get('experiment_name', 'imdb-bert'),
                config=config
            )
        
        # Load data
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            dataset_name=config['dataset'],
            model_type='transformer',
            tokenizer_name=config.get('tokenizer_name', 'bert-base-uncased'),
            max_length=config.get('max_length', 512),
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4)
        )
        
        # Initialize model
        self.model = get_model(
            config['model_name'],
            num_classes=config['num_classes']
        ).to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.get('label_smoothing', 0.0)
        )
        
        # Optimizer
        self.optimizer = self._get_optimizer()
        
        # Scheduler
        self.scheduler = self._get_scheduler()
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 10),
            min_delta=config.get('early_stopping_min_delta', 0.001)
        )
        
        # Training state
        self.start_epoch = 0
        self.best_acc = 0.0
        self.best_f1 = 0.0
        
        # Resume from checkpoint if specified
        if config.get('resume'):
            self.load_checkpoint(config['resume'])
    
    def _get_optimizer(self):
        """Get optimizer based on config"""
        opt_name = self.config.get('optimizer', 'adamw').lower()
        lr = self.config['learning_rate']
        weight_decay = self.config.get('weight_decay', 0.01)
        
        # Different learning rates for different parts
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        
        if opt_name == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
        elif opt_name == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=lr)
        elif opt_name == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
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
        elif scheduler_name == 'reduce':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        else:
            return None
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    if hasattr(self.model, 'forward'):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                        elif hasattr(outputs, 'logits'):
                            logits = outputs.logits
                        else:
                            logits = outputs
                        loss = self.criterion(logits, labels)
                    else:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                        logits = outputs.logits
            else:
                if hasattr(self.model, 'forward'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    elif hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                    loss = self.criterion(logits, labels)
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    logits = outputs.logits
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            
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
            losses.update(loss.item(), input_ids.size(0))
            accuracies.update(acc.item(), input_ids.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{accuracies.avg:.4f}'
            })
        
        return {
            'loss': losses.avg,
            'accuracy': accuracies.avg
        }
    
    def validate(self, loader=None) -> Dict:
        """Validate the model"""
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        
        losses = AverageMeter()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    elif hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                    loss = self.criterion(logits, labels)
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    logits = outputs.logits
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                
                losses.update(loss.item(), input_ids.size(0))
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        return {
            'loss': losses.avg,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.start_epoch, self.config['epochs']):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            print(f"\nEpoch {epoch}/{self.config['epochs']}:")
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"Val F1: {val_metrics['f1']:.4f}")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_acc': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_acc
            if is_best:
                self.best_acc = val_metrics['accuracy']
                self.best_f1 = val_metrics['f1']
            
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_acc': self.best_acc,
                'best_f1': self.best_f1,
                'config': self.config
            }, is_best, checkpoint_dir=self.config['checkpoint_dir'])
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                print("Early stopping triggered!")
                break
        
        # Final test evaluation
        print("\nEvaluating on test set...")
        test_metrics = self.validate(self.test_loader)
        print(f"Test Loss: {test_metrics['loss']:.4f}, "
              f"Test Acc: {test_metrics['accuracy']:.4f}, "
              f"Test F1: {test_metrics['f1']:.4f}")
        
        if self.config.get('use_wandb', False):
            wandb.log({
                'test_loss': test_metrics['loss'],
                'test_acc': test_metrics['accuracy'],
                'test_f1': test_metrics['f1'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall']
            })
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = load_checkpoint(checkpoint_path, self.model, self.optimizer)
        self.start_epoch = checkpoint.get('epoch', 0)
        self.best_acc = checkpoint.get('best_acc', 0)
        self.best_f1 = checkpoint.get('best_f1', 0)
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.start_epoch}")


def main():
    parser = argparse.ArgumentParser(description='Text Classification Training')
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='bert',
                        help='Model architecture')
    parser.add_argument('--dataset', type=str, default='imdb',
                        help='Dataset name')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint')
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override config
    config['model_name'] = args.model
    config['dataset'] = args.dataset
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['resume'] = args.resume
    
    # Set defaults
    config.setdefault('num_classes', 2 if args.dataset == 'imdb' else 4)
    config.setdefault('checkpoint_dir', './checkpoints')
    config.setdefault('num_workers', 4)
    config.setdefault('max_length', 512 if args.dataset == 'imdb' else 256)
    config.setdefault('use_amp', True)
    config.setdefault('use_wandb', False)
    config.setdefault('label_smoothing', 0.1)
    config.setdefault('weight_decay', 0.01)
    config.setdefault('gradient_clip', 1.0)
    config.setdefault('optimizer', 'adamw')
    config.setdefault('scheduler', 'cosine')
    config.setdefault('early_stopping_patience', 10)
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Train
    trainer = TextClassificationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()