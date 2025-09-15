"""
Training script for Speech Emotion Recognition
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
from pathlib import Path
from typing import Dict, List, Tuple
import wandb
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import EmotionDataModule, collate_fn
from models import get_model
from utils import EarlyStopping, plot_confusion_matrix, save_training_curves


class EmotionTrainer:
    """Trainer for emotion recognition models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize data module
        self.data_module = EmotionDataModule(
            dataset_name=config['dataset'],
            data_path=config['data_path'],
            batch_size=config['batch_size'],
            feature_type=config['feature_type'],
            num_workers=config.get('num_workers', 4)
        )
        self.data_module.setup()
        
        # Get input dimension based on feature type
        input_dim = self._get_input_dim()
        
        # Initialize model
        self.model = get_model(
            config['model_name'],
            input_dim=input_dim,
            num_classes=self.data_module.num_classes,
            dropout=config.get('dropout', 0.5)
        ).to(self.device)
        
        print(f"Model: {config['model_name']}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
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
            mode='max',
            delta=0.001
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project="speech-emotion-recognition",
                config=config,
                name=f"{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def _get_input_dim(self) -> int:
        """Get input dimension based on feature type"""
        feature_type = self.config['feature_type']
        
        if feature_type == 'raw':
            return 1
        elif feature_type == 'mfcc':
            return 120  # 40 MFCCs + 40 deltas + 40 delta-deltas
        elif feature_type == 'melspec':
            return 128  # 128 mel bands
        elif feature_type == 'combined':
            return 220  # Combined features
        else:
            return 40  # Default
    
    def _get_optimizer(self):
        """Get optimizer"""
        opt_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config['learning_rate']
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if opt_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_name == 'adamw':
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
        elif scheduler_name == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        else:
            return None
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        train_loader = self.data_module.train_dataloader()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
            
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
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc:.2f}%'
            })
            
            # Log to wandb
            if self.config.get('use_wandb', False) and batch_idx % 10 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'train_acc': acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self) -> Tuple[float, float, Dict]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        val_loader = self.data_module.val_dataloader()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = np.mean(all_preds == all_labels) * 100
        
        # Get classification report
        class_names = self.data_module.label_encoder.classes_
        report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            output_dict=True
        )
        
        return avg_loss, accuracy, {
            'predictions': all_preds,
            'labels': all_labels,
            'report': report
        }
    
    def train(self):
        """Main training loop"""
        print("\nStarting training...")
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validation
            val_loss, val_acc, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.config['epochs']}:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                
                # Save confusion matrix
                cm = confusion_matrix(
                    val_metrics['labels'],
                    val_metrics['predictions']
                )
                plot_confusion_matrix(
                    cm,
                    self.data_module.label_encoder.classes_,
                    save_path=os.path.join(self.config['checkpoint_dir'], 'confusion_matrix.png')
                )
            
            # Early stopping
            if self.early_stopping(val_acc):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        # Save training curves
        save_training_curves(
            self.train_losses,
            self.val_losses,
            self.train_accs,
            self.val_accs,
            save_path=os.path.join(self.config['checkpoint_dir'], 'training_curves.png')
        )
        
        print(f"\nTraining completed. Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def test(self):
        """Test the model"""
        print("\nTesting model...")
        
        # Load best model
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        test_loader = self.data_module.test_dataloader()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(features)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = np.mean(all_preds == all_labels) * 100
        
        # Get classification report
        class_names = self.data_module.label_encoder.classes_
        report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names
        )
        
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        print("\nClassification Report:")
        print(report)
        
        # Save test results
        results = {
            'test_accuracy': accuracy,
            'classification_report': report,
            'config': self.config
        }
        
        with open(os.path.join(self.config['checkpoint_dir'], 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with accuracy: {self.best_val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition Training')
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Path to config file')
    parser.add_argument('--dataset', type=str, default='ravdess',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='crnn',
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--feature-type', type=str, default='mfcc',
                        choices=['raw', 'mfcc', 'melspec', 'combined'],
                        help='Feature type')
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override with command line arguments
    config.update({
        'dataset': args.dataset,
        'model_name': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'feature_type': args.feature_type,
        'data_path': config.get('data_path', './data'),
        'checkpoint_dir': config.get('checkpoint_dir', './checkpoints'),
        'optimizer': config.get('optimizer', 'adam'),
        'scheduler': config.get('scheduler', 'cosine'),
        'dropout': config.get('dropout', 0.5),
        'gradient_clip': config.get('gradient_clip', 1.0),
        'label_smoothing': config.get('label_smoothing', 0.1),
        'use_amp': config.get('use_amp', True),
        'early_stopping_patience': config.get('early_stopping_patience', 10),
        'num_workers': config.get('num_workers', 4),
        'use_wandb': config.get('use_wandb', False)
    })
    
    # Create trainer and train
    trainer = EmotionTrainer(config)
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    main()